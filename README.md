
# Transformer 模型教程

本仓库包含一个Jupyter Notebook: [Transformer教程.ipynb](transformer教程.ipynb)，演示了如何使用Transformer模型，例如GPT-2，进行文本分类任务，使用了Hugging Face的Transformers库。

## 内容

- Transformer模型的介绍和背景
- 模型加载和初始设置
- 使用GPT-2进行文本分类的示例
- 输出结果的解释
- 可视化展示模型结构和训练过程

![Intro](./238353467-897cd757-ea1f-492d-aaf9-6d1674177e08.gif)

## Requirement

- Python 3.6+
- Jupyter Notebook
- Hugging Face Transformers 库


## 示例

以下是教程中一个使用GPT-2分类单个句子的示例：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# 指定模型和分词器
model_name = "gpt2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建文本分类管道
text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# 对单个句子进行分类
result = text_classifier("I love machine learning!")

# 输出结果
print(result)
```

## 联系方式

如有任何问题或反馈，请联系我: lidaoyuan2816@gmail.com。
