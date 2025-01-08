import os
import copy
from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gc
import ast
from tqdm import tqdm
import json
import shutil
import sys
from glob import glob

import polars as pl
import polars
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

# from datasets import Dataset, DatasetDict, load_dataset
import transformers
import datasets
import sentence_transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,

    PreTrainedTokenizerFast,
    PreTrainedTokenizerBase, 
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator

# os.environ['NCCL_P2P_DISABLE'] = '1' # RTX 4000 doesn't support
# os.environ['NCCL_IB_DISABLE'] = '1' # RTX 4000 doesn't support

from utils import seed_torch, current_date_time, get_timediff
from utils import load_yaml, simple_namespace, write_to_summary_log, init_logger
from utils import mapk, last_token_pool
from loss_utils import compute_no_in_batch_neg_loss

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='train_retriever_v0.yaml')
parser.add_argument('--rank', type=str, default="0,1")
args = parser.parse_args()

cfg = load_yaml(args.cfg)
cfg = simple_namespace(cfg)
if args.rank:
    cfg.general.rank = args.rank
print(f"cfg.general.rank: {cfg.general.rank}")
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.rank
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

base_dir = "."
input_dir = f"{base_dir}/input"
comp_dir = f"{input_dir}/eedi-mining-misconceptions-in-mathematics"
output_dir = f"{base_dir}/output"
summary_log_path = f"{output_dir}/summary_retriever.log"

seed_torch(cfg.general.seed)
cur_time = current_date_time()
cur_time_abbr = cur_time.replace("-", "").replace(":", "").replace(" ", "")[4:12]
output_dir = f"{output_dir}/{cur_time_abbr}_retriever"
os.makedirs(output_dir, exist_ok=True)
LOGGER = init_logger(f'{output_dir}/train.log')
shutil.copy(args.cfg, f"{output_dir}/{args.cfg}")


num_gpus = torch.cuda.device_count()
LOGGER.info(f"可用的 GPU 数量: {num_gpus}")

if cfg.general.report_to == "wandb":
    import wandb
    wandb.login()
    wandb.init(project=f"{cfg.model.model_name.split('/')[-1]}", name=cur_time_abbr)


LOGGER.info(f"polars=={polars.__version__}")
LOGGER.info(f"torch=={torch.__version__}")
LOGGER.info(f"transformers=={transformers.__version__}")
LOGGER.info(f"datasets=={datasets.__version__}")
LOGGER.info(f"sentence_transformers=={sentence_transformers.__version__}")
LOGGER.info(f"")

# %% ==================  Read data =======================
if os.path.exists(f"{comp_dir}/{cfg.data.long_df_pq}"):
    long_df = pd.read_parquet(f"{comp_dir}/{cfg.data.long_df_pq}")
    LOGGER.info(f"load long_df, explode_df from parquet file.")
    misconception_mapping_df = pl.read_csv(f"{comp_dir}/misconception_mapping.csv")
    misconception_name = misconception_mapping_df["MisconceptionName"].to_list()
    misconception_dict = misconception_mapping_df.to_pandas().set_index('MisconceptionId')['MisconceptionName'].to_dict()
    LOGGER.info(f"len(misconception_mapping_df): {len(misconception_mapping_df)}")
else:
    train_df = pl.read_csv(f"{comp_dir}/train_folds.csv")
    LOGGER.info(f"len(train_df): {len(train_df)}")
    misconception_mapping_df = pl.read_csv(f"{comp_dir}/misconception_mapping.csv")
    misconception_name = misconception_mapping_df["MisconceptionName"].to_list()
    misconception_dict = misconception_mapping_df.to_pandas().set_index('MisconceptionId')['MisconceptionName'].to_dict()
    LOGGER.info(f"len(misconception_mapping_df): {len(misconception_mapping_df)}")
    LOGGER.info(f"")

    # 定义常用的列名列表
    common_col = [
        "QuestionId",
        "ConstructName",
        "SubjectName",
        "QuestionText",
        "CorrectAnswer",
        "fold",
    ]

    # 对训练集数据进行处理，转换为长表格式，并添加需要的列
    long_df = (
        train_df
        # 选择需要的列，包括common_col和所有的Answer[A-D]Text列
        .select(
            pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]])
        )
        # 获取 CorrectAnswer 的 Text，创建新列 CorrectAnswerText
        .with_columns(
            pl.when(pl.col("CorrectAnswer") == "A").then(pl.col("AnswerAText"))
            .when(pl.col("CorrectAnswer") == "B").then(pl.col("AnswerBText"))
            .when(pl.col("CorrectAnswer") == "C").then(pl.col("AnswerCText"))
            .when(pl.col("CorrectAnswer") == "D").then(pl.col("AnswerDText"))
            .otherwise(None)
            .alias("CorrectAnswerText")
        )
        # 使用unpivot函数将宽表转换为长表，将Answer[A-D]Text列展开
        .unpivot(
            index=common_col+["CorrectAnswerText"], # 保持这些列不变
            variable_name="AnswerType", # 展开列的名称存储在新列AnswerType中
            value_name="AnswerText",    # 展开列的值存储在新列AnswerText中
        )
        # 添加新列
        .with_columns(
            # 将ConstructName、SubjectName、QuestionText和AnswerText列拼接成一个字符串，存储在AllText列中
            pl.concat_str(
                [
                    '### Construct\n' +  pl.col("ConstructName"),
                    '\n### Subject\n' + pl.col("SubjectName"),
                    '\n### Question\n'+ pl.col("QuestionText"),
                    '\n### Correct Answer\n' + pl.col("CorrectAnswerText"),
                    '\n### Wrong Answer\n' + pl.col("AnswerText"),
                ],
                separator="",
            ).alias("AllText"),
            # 从AnswerType列中提取选项字母（A-D），存储在AnswerAlphabet列中
            pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
        )
        # 创建QuestionId_Answer列，将QuestionId和AnswerAlphabet拼接，形成唯一标识
        .with_columns(
            pl.concat_str(
                [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
            ).alias("QuestionId_Answer"),
        )
        # 按照QuestionId_Answer进行排序
        .sort("QuestionId_Answer")
    )

    # 对误解映射数据进行处理，转换为长表格式，并添加需要的列
    misconception_mapping_df_long = (
        train_df.select(
            # 选择需要的列，包括common_col和所有的Misconception[A-D]Id列
            pl.col(
                common_col + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]
            )
        )
        # 使用unpivot函数将宽表转换为长表，将Misconception[A-D]Id列展开
        .unpivot(
            index=common_col,                # 保持这些列不变
            variable_name="MisconceptionType", # 展开列的名称存储在MisconceptionType中
            value_name="MisconceptionId",      # 展开列的值存储在MisconceptionId中
        )
        # 从MisconceptionType列中提取选项字母（A-D），存储在AnswerAlphabet列中
        .with_columns(
            pl.col("MisconceptionType")
            .str.extract(r"Misconception([A-D])Id$")
            .alias("AnswerAlphabet"),
        )
        # 创建QuestionId_Answer列，将QuestionId和AnswerAlphabet拼接，形成唯一标识
        .with_columns(
            pl.concat_str(
                [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
            ).alias("QuestionId_Answer"),
        )
        # 按照QuestionId_Answer进行排序
        .sort("QuestionId_Answer")
        # 选择需要的列
        .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
        # 将MisconceptionId列的数据类型转换为Int64
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )

    long_df = long_df.join(misconception_mapping_df_long, on="QuestionId_Answer")
    LOGGER.info(f"long_df len (has nan): {len(long_df)}")

    # =================== Dataset ====================
    last_oof_df = pd.read_csv(f"{input_dir}/{cfg.data.oof_csv}")[["QuestionId_Answer", "preds_all_mm_ids"]]
    last_oof_df["preds_all_mm_ids"] = last_oof_df["preds_all_mm_ids"].apply(ast.literal_eval)
    last_oof_df["pred_mm_id"] = last_oof_df["preds_all_mm_ids"].apply(lambda x: x[:cfg.data.top_nums])
    last_oof_df = pl.DataFrame(last_oof_df)

    long_df = long_df.join(last_oof_df, on="QuestionId_Answer", how="left")


    long_df = long_df.to_pandas()
    LOGGER.info(f"{long_df.columns = }")
    # 只选择 MisconceptionId 不为NaN 的数据
    long_df = long_df[~pd.isna(long_df["MisconceptionId"])].reset_index(drop=True)
    long_df["MisconceptionId"] = long_df["MisconceptionId"].astype(int)
    long_df = long_df[["QuestionId_Answer", "AllText", "MisconceptionId", "preds_all_mm_ids", "fold"]]
    LOGGER.info(f"long_df shape (after del nan): {long_df.shape}")
    # 保存数据
    long_df.to_parquet(f"{comp_dir}/{cfg.data.long_df_pq}")



def adjust_passage_ids(row, pred_col, topk=25):
    misconception_id = row['MisconceptionId']
    
    if isinstance(row[pred_col], list):
        predict_list = row[pred_col]
    else:
        predict_list = row[pred_col].tolist()
    
    # 如果MisconceptionId在 preds_all_mm_ids 中，调整到最前面
    if misconception_id in predict_list:
        predict_list.remove(misconception_id)
        predict_list.insert(0, misconception_id)
    else:
        # 如果不在，插入到最前面并去掉最后一个元素
        predict_list.insert(0, misconception_id)
        predict_list.pop()
    
    predict_list = predict_list[:topk]
    
    return predict_list

def convert_to_text(passages_list):
    return [misconception_dict.get(misconception_id, '') for misconception_id in passages_list]

long_df['passage_ids'] = long_df.apply(lambda row: adjust_passage_ids(row, pred_col="preds_all_mm_ids", topk=cfg.data.top_nums), axis=1)
long_df['passage_texts'] = long_df['passage_ids'].apply(convert_to_text)

# AllText token len: max:404, median: 91
# MisconceptionName token len: max: 45, median: 14
# pred_mm_name token len: max: 45, median: 14

if cfg.data.full_train_data:
    tra_long_df = long_df
    val_long_df = long_df
    LOGGER.info(f"full tra_long_df, len(long_df): {len(long_df)}")
else:
    tra_long_df = long_df[long_df["fold"] != cfg.data.fold_idx].reset_index(drop=True)
    val_long_df = long_df[long_df["fold"] == cfg.data.fold_idx].reset_index(drop=True)
    LOGGER.info(f"len(tra_long_df): {len(tra_long_df)}, len(val_long_df): {len(val_long_df)}, val rate: {len(val_long_df) / (len(tra_long_df)+len(val_long_df)):.1%}")


# %% ========= Tokenizer and Dataset ========= 
def add_suffix(text, suffix_text, is_query):
    text = f"{suffix_text}{text}"
    text = text.strip()
    if is_query:
        text = f"{text}\n<response>"
    return text

class QPDataset(Dataset):
    def __init__(self, tra_long_df, shuffle=True):
        # train_df_long to data
        if cfg.general.debug:
            tra_long_df = tra_long_df.sample(frac=cfg.general.debug_size, random_state=cfg.general.seed).reset_index(drop=True)
            LOGGER.info(f"debug mode, len(tra_long_df): {len(tra_long_df)}")

        self.queries = tra_long_df['AllText'].tolist()
        self.passages = tra_long_df['passage_texts'].tolist()

        if shuffle:
            list_len = len(self.queries)
            indices = np.arange(list_len)
            np.random.shuffle(indices)
            self.queries = [self.queries[i] for i in indices]
            self.passages = [self.passages[i] for i in indices]

        self.queries = [add_suffix(x, cfg.data.query_prefix, is_query=True) for x in self.queries]
        self.passages = [
            [add_suffix(x, cfg.data.mis_prefix, is_query=False) for x in passage]
            for passage in self.passages
        ]
        assert len(self.queries) == len(self.passages), f"{len(self.queries) = } != {len(self.passages) = }"

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            'queries': self.queries[idx],
            'passages': self.passages[idx],
        }

def collate_fn(batch):
    queries =  [item['queries'] for item in batch]
    passages = [item['passages'] for item in batch]
    return {
        'queries': queries,
        'passages': passages
    }

if cfg.data.peek_dataset:
    tmp_qp_dataset = QPDataset(tra_long_df)
    one_sample = tmp_qp_dataset[0]
    LOGGER.info(f"\nQUERY:\n{one_sample['queries']}")
    LOGGER.info(f"\nGT:\n{one_sample['passages'][0]}")
    LOGGER.info(f"\nPASSAGES:\n{one_sample['passages']}")


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"{input_dir}/{cfg.model.model_name}", trust_remote_code=True)

# %% ========= Model =========
layers_num = 42
lora_config = LoraConfig(
    r=cfg.model.lora_r,
    lora_alpha=cfg.model.lora_alpha,
    lora_dropout=cfg.model.lora_dropout,
    bias=cfg.model.lora_bias,
    task_type=TaskType.FEATURE_EXTRACTION, # TaskType.FEATURE_EXTRACTION TaskType.CAUSAL_LM
    target_modules=cfg.model.lora_target_modules,
    # layers_to_transform=[i for i in range(layers_num) if i >= cfg.model.freeze_layers],
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if cfg.training.amp=="bf16" else torch.float16,
    )

model = AutoModel.from_pretrained(
    f"{input_dir}/{cfg.model.model_name}",
    torch_dtype=torch.bfloat16 if cfg.training.amp=="bf16" else torch.float16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    # attn_implementation="flash_attention_2",
)

model.config.use_cache = False

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# %% ========= Training =========
def valid_func(model, tokenizer, df, query_prefix, mis_prefix, mode="valid"):
    model.eval()
    batch_size = cfg.training.per_device_eval_batch_size

    query_list = df["AllText"].to_list()
    query_result = []
    for i in tqdm(range(0, len(query_list), batch_size)):
        batch_query_list = query_list[i:i+batch_size]
        batch_query_list = [add_suffix(x, query_prefix, is_query=True) for x in batch_query_list]
        query_encodings = tokenizer(
            batch_query_list,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=cfg.model.query_max_length,
        )
        input_ids = query_encodings['input_ids'].to(model.device)
        attention_mask = query_encodings['attention_mask'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) 
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1) # shape: (4370, 4096)
        query_result.append(embeddings)
        torch.cuda.empty_cache()
    query_embeddings = torch.cat(query_result, dim=0)


    misconception_result = []
    for i in tqdm(range(0, len(misconception_name), batch_size)):
        batch_misconception_name = misconception_name[i:i+batch_size]
        batch_misconception_name = [add_suffix(x, mis_prefix, is_query=False) for x in batch_misconception_name]
        misconception_encodings = tokenizer(
                    batch_misconception_name,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=cfg.model.mis_max_length,
                )
        input_ids = misconception_encodings['input_ids'].to(model.device)
        attention_mask = misconception_encodings['attention_mask'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) 
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1) # shape: (2587, 4096)
        misconception_result.append(embeddings)
        torch.cuda.empty_cache()
    misconception_embeddings = torch.cat(misconception_result, dim=0)


    scores = (query_embeddings @ misconception_embeddings.T) * 100 # shape: (len(df), 2587)
    scores = scores.float()
    scores = scores.cpu().numpy()
    LOGGER.info(f"{scores.shape = }")
    # 获取误解id的index,按照score排序
    preds_all_mm_ids = np.argsort(-scores, axis=1)
    preds_top25_mm_ids = preds_all_mm_ids[:, :25]

    df["preds_all_mm_ids"] = preds_all_mm_ids.tolist()
    df["preds_top25_mm_ids"] = preds_top25_mm_ids.tolist()

    return df




def encode_texts(model, tokenizer, texts, max_length):
    if type(texts[0]) == list:
        # shape: (batch_size, group_size) -> (batch_size * group_size)
        texts = [text for texts_ in texts for text in texts_]
    # 进行分词
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_length,
    )
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # 获取嵌入表示
    # padding_side='left'
    embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
    # embeddings = outputs[0][:, -1, :]
    return embeddings



def train_func(model, val_long_df):
    train_dataset = QPDataset(tra_long_df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    accelerator = Accelerator()
    device = accelerator.device

    if cfg.training.optim_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        total_steps=cfg.training.n_epochs * len(train_dataloader) // cfg.training.gradient_accumulation_steps,
        pct_start=cfg.training.one_cycle_pct_start,
        anneal_strategy='cos', 
        div_factor=25.0,
        final_div_factor=100,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    losses = []
    lrs = []
    for epoch in range(cfg.training.n_epochs):
        model.train()  
        step = 0
        bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{cfg.training.n_epochs}', total=len(train_dataloader), ncols=100)
        for batch_idx, batch in enumerate(bar):
            step += 1

            queries = batch['queries']
            passages = batch['passages']

            # 编码文本 
            queries_embeddings = encode_texts(model, tokenizer, queries, max_length=cfg.model.query_max_length)
            passages_embeddings = encode_texts(model, tokenizer, passages, max_length=cfg.model.mis_max_length)

            queries_embeddings = F.normalize(queries_embeddings, p=2, dim=1)
            passages_embeddings = F.normalize(passages_embeddings, p=2, dim=1)
            # queries_embeddings.shape = [4, 2048]
            # passages_embeddings.shape = [60, 2048] , 每个query对应15个passage, 其中第0个是正确答案, 其余14个是错误答案

            local_scores, loss = compute_no_in_batch_neg_loss(queries_embeddings, passages_embeddings, temperature=cfg.training.temperature)            
            loss = loss / cfg.training.gradient_accumulation_steps
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

            bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        avg_loss = np.mean(losses[-10:])
        print(f"Epoch {epoch+1}/{cfg.training.n_epochs}, Average Loss: {avg_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

        # Valid
        train_oof_df = valid_func(model, tokenizer, tra_long_df, cfg.data.query_prefix, cfg.data.mis_prefix, mode="train")
        oof_df = valid_func(model, tokenizer, val_long_df, cfg.data.query_prefix, cfg.data.mis_prefix, mode="valid")

        mapk_score, recall_scores = get_result(oof_df)
        log_str = f"MAP@25: {mapk_score:.4f}\n"
        log_str += " | ".join([f"R@{k}: {recall_scores[f'recall@{k}']:.4f}" for k in [1, 10, 25, 50, 100]])
        LOGGER.info(log_str)

    # 画出损失曲线
    move_windows = 3
    ma_losses = pd.Series(losses).rolling(window=move_windows, min_periods=1).mean()
    plt.plot(ma_losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f"{output_dir}/training_loss.png")

    # 清空之前的plt
    plt.clf()
    
    # 画出学习率曲线
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.savefig(f"{output_dir}/learning_rate.png")

    # save model to disk
    model.save_pretrained(f"{output_dir}/{cur_time_abbr}adapetermodel")
    tokenizer.save_pretrained(f"{output_dir}/{cur_time_abbr}adapetermodel")
    LOGGER.info("save adapeter model.")

    return train_oof_df, oof_df


# ======================= Run ==========================
def get_result(oof_df):
    # Compute MAP@25
    label = oof_df["MisconceptionId"].tolist()
    label = [[i] for i in label]
    preds = oof_df["preds_top25_mm_ids"].tolist()
    mapk_score = mapk(label, preds)

    # Compute recalls at various cutoffs
    ks = [1, 10, 25, 50, 100]
    recall_scores = {}
    ground_truth_ids = oof_df["MisconceptionId"].tolist()
    all_predictions = oof_df["preds_all_mm_ids"].tolist()  # Each is a list of predicted MisconceptionIds
    
    for k in ks:
        num_correct = sum([1 if gt_id in preds[:k] else 0 for gt_id, preds in zip(ground_truth_ids, all_predictions)])
        recall = num_correct / len(oof_df)
        recall_scores[f"recall@{k}"] = recall
    return mapk_score, recall_scores

start_time = time.time()
train_oof_df, oof_df = train_func(model, val_long_df)
train_oof_df.to_csv(f"{output_dir}/train_oof_df.csv", index=False)
oof_df.to_csv(f"{output_dir}/oof_df.csv", index=False)

mapk_score, recall_scores = get_result(oof_df)


result_log = f"MAP@25: {mapk_score:.4f}\n"
result_log += " | ".join([f"R@{k}: {recall_scores[f'recall@{k}']:.4f}" for k in [1, 10, 25, 50, 100]])
result_log += f"\nElapsed time: {get_timediff(start_time, time.time())}"

LOGGER.info(result_log)
write_to_summary_log(summary_log_path, result_log)