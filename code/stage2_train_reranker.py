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
# from torch.utils.data import Dataset, DataLoader
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
from datasets import Dataset, DatasetDict, load_dataset

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

# os.environ['NCCL_P2P_DISABLE'] = '1' # RTX 4000 doesn't support
# os.environ['NCCL_IB_DISABLE'] = '1' # RTX 4000 doesn't support

from utils import seed_torch, current_date_time, get_timediff
from utils import load_yaml, simple_namespace, write_to_summary_log, init_logger
from utils import mapk, last_token_pool


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='train_reranker_v0.yaml')
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
summary_log_path = f"{output_dir}/summary_reranker.log"

seed_torch(cfg.general.seed)
cur_time = current_date_time()
cur_time_abbr = cur_time.replace("-", "").replace(":", "").replace(" ", "")[4:12]
output_dir = f"{output_dir}/{cur_time_abbr}_reranker"
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
misconception_mapping_df = pl.read_csv(f"{comp_dir}/misconception_mapping.csv")
misconception_name = misconception_mapping_df["MisconceptionName"].to_list()
misconception_dict = misconception_mapping_df.to_pandas().set_index('MisconceptionId')['MisconceptionName'].to_dict()

train_oof_csv = pd.read_csv(f"{cfg.data.oof_dir}/train_oof_df.csv")[["QuestionId_Answer", "AllText", "MisconceptionId", "preds_all_mm_ids"]]
valid_oof_csv = pd.read_csv(f"{cfg.data.oof_dir}/oof_df.csv")[["QuestionId_Answer", "AllText", "MisconceptionId", "preds_all_mm_ids"]]

def data_preprocess(df, is_train):
    df["preds_all_mm_ids"] = df["preds_all_mm_ids"].apply(lambda x: ast.literal_eval(x))
    df["top_mm_ids"] = df["preds_all_mm_ids"].apply(lambda x: x[:5])
    # 如果 MisconceptionId 不存在于 top_mm_ids 中, 则将MisconceptionId替换列表中最后一个值
    df["top_mm_ids"] = df.apply(
        lambda row: row["top_mm_ids"] if row["MisconceptionId"] in row["top_mm_ids"] else row["top_mm_ids"][:-1] + [row["MisconceptionId"]],
        axis=1
    )
    # 对 top_mm_ids 的顺序洗牌
    df["top_mm_ids"] = df["top_mm_ids"].apply(lambda x: np.random.permutation(x).tolist())

    # 新建一列 gt_idx, 表示MisconceptionId在top_mm_ids中的位置
    df["gt_idx"] = df.apply(lambda row: row["top_mm_ids"].index(row["MisconceptionId"]), axis=1)
    i2l_dict = {0: "A", 1: "B", 2: "C", 3:"D", 4:"E"}
    df["gt"] = df["gt_idx"].apply(lambda x: i2l_dict[x])

    # 新建一列 top_mm_texts, 也是一个列表, 其中的值是top_mm_ids对应于misconception_dict中的value
    df["top_mm_texts"] = df["top_mm_ids"].apply(lambda ids: [misconception_dict[id] for id in ids])

    # 在 AllText 后面加上新"\n\nHere are 5 possible candidates for misconception:\n"
    df["AllText"] = df["AllText"] + "\n\nHere are 5 possible candidates for misconception:\n"

    # 在 AllText 后面加上5个候选项,候选项来自top_mm_texts, 然后要这样的格式 "A. candidate0\nB. candidate1\nC. candidate2\nD. candidate3\nE. candidate4"
    df["AllText"] = df.apply(
        lambda row: row["AllText"] + "\n".join([f"{chr(65+i)}. {candidate}" for i, candidate in enumerate(row["top_mm_texts"])]),
        axis=1
    )

    # 在 AllText 后面加上新"\nWhich misconception candidate best explains what led to the wrong answer? (Please directly answer A, B, C, D or E)"
    df["AllText"] = df["AllText"] + "\nWhich misconception candidate best explains what led to the wrong answer? (Please directly answer A, B, C, D or E)\nAnswer:"

    return df


train_oof_csv = data_preprocess(train_oof_csv, is_train=True)
valid_oof_csv = data_preprocess(valid_oof_csv, is_train=False)



# %% ========= Tokenizer and Dataset ========= 
model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    # unsloth/Qwen2.5-1.5B-Instruct
    model_name = cfg.model.model_name,
    max_seq_length = cfg.model.max_length,
    dtype = None,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

PROMPT  = """<|im_start|>system
Given a math question and its incorrect answer, identify the underlying misconception that led to the mistake.<|im_end|>
<|im_start|>user
{AllText}<|im_end|>
<|im_start|>assistant
{AnswerLetter}<|im_end|>
"""

# 定义函数，用于将每一行数据填充到模板中
def apply_template(row):
    instruction_text =  PROMPT.format(
        AllText=row["AllText"],
        AnswerLetter=row["gt"],
    )

    return instruction_text

train_oof_csv[["instruction"]] = train_oof_csv.apply(lambda row: pd.Series(apply_template(row)), axis=1)
valid_oof_csv[["instruction"]] = valid_oof_csv.apply(lambda row: pd.Series(apply_template(row)), axis=1)

train_oof_csv["instruction_token_len"] = train_oof_csv["instruction"].apply(lambda x: len(tokenizer(x)["input_ids"]))
valid_oof_csv["instruction_token_len"] = valid_oof_csv["instruction"].apply(lambda x: len(tokenizer(x)["input_ids"]))
LOGGER.info(f"train instruction_token_len range: {train_oof_csv['instruction_token_len'].min()} ~ {train_oof_csv['instruction_token_len'].max()}")
LOGGER.info(f"valid instruction_token_len range: {valid_oof_csv['instruction_token_len'].min()} ~ {valid_oof_csv['instruction_token_len'].max()}")

dataset = Dataset.from_pandas(train_oof_csv)
data = DatasetDict({"train": dataset})

# %% ========= Model =========
model = FastLanguageModel.get_peft_model(
    model,
    r = cfg.model.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = cfg.model.lora_target_modules,
    lora_alpha = cfg.model.lora_alpha,
    lora_dropout = cfg.model.lora_dropout, # Supports any, but = 0 is optimized
    bias = cfg.model.lora_bias,    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = cfg.general.seed,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

LOGGER.info(f"model:\n{model}\n\n")

# %% ========= Training =========
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "instruction",
    max_seq_length = cfg.model.max_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = cfg.general.num_workers,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.training.gradient_accumulation_steps,
        warmup_steps = cfg.training.warmup_steps,
        num_train_epochs=cfg.training.n_epochs, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = cfg.training.lr,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = cfg.training.optim_type,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = cfg.general.seed,
        output_dir = output_dir,
        report_to = cfg.general.report_to,
    ),
)

# 只训练response部分
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# 开始训练
LOGGER.info("start training...")
trainer_stats = trainer.train()

# 保存adapt模型
os.makedirs("lora_model", exist_ok=True)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
# trainer.save_model(f"{output_dir}/{cur_time_abbr}adapetermodel")
LOGGER.info("finish training...")

# save 4bit for vllm
# model.save_pretrained_merged("model_4bit", tokenizer, save_method = "merged_4bit_forced",)
# LOGGER.info("finish saving 4bit model...")

# 记录 loss
LOGGER.info(f"Training loss: {trainer_stats.training_loss}")
write_to_summary_log(summary_log_path,  f"Training loss: {trainer_stats.training_loss}")