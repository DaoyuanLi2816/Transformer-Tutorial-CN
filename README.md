# Eedi - Mining Misconceptions in Mathematics Solution

This solution was developed for the [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics) competition on Kaggle, where participants were challenged to create models to predict the affinity between incorrect options (distractors) and potential misconceptions in mathematics multiple-choice questions. The task required building a model capable of recommending candidate misconceptions for each incorrect option to assist education experts in more efficiently and consistently labeling misconceptions.

Our team achieved a [Silver Medal](https://www.kaggle.com/certification/competitions/distiller/eedi-mining-misconceptions-in-mathematics), with a public score of **0.54** and a private score of **0.50**. Our solution showcased the potential for using machine learning and natural language processing models, such as Qwen2.5-32B-Instruct combined with LoRA fine-tuning, to improve the efficiency and accuracy of misconception labeling in education, contributing to advancements in educational AI. 🥈

![Daoyuan Li - Eedi - Mining Misconceptions in Mathematics](./Certificate.png)

## Competition Overview

### Competition Introduction

This competition aims to develop a machine-learning-based natural language processing (NLP) model that can accurately predict the affinity between incorrect options (distractors) and potential misconceptions in mathematics multiple-choice questions. The model will recommend candidate misconceptions for each incorrect option, assisting education experts in labeling misconceptions more efficiently and consistently.

### Competition Background

In mathematics education, diagnostic questions (DQs) typically contain one correct answer and three carefully designed distractors, each corresponding to a specific student misconception. For example, if a student selects the incorrect option "13," it might indicate a misconception of "ignoring the order of operations and calculating from left to right sequentially."

Manually labeling misconceptions for each distractor is time-consuming and prone to inconsistency. Furthermore, new misconceptions may emerge as knowledge areas expand. Therefore, developing a model that can automatically recommend misconceptions is crucial.

## Solution Overview

This solution comprises two stages:

1. **Retriever Stage**: Utilize a retrieval model to recommend candidate misconceptions for each incorrect option.
2. **Reranker Stage**: Re-rank the top 5 candidate misconceptions recommended by the Retriever to improve recommendation accuracy.

The competition uses **Mean Average Precision @ 25 (MAP@25)** as the evaluation metric, which calculates the average precision of predicted misconception lists for each sample and then averages them across all samples.

## Detailed Solution

### 1. Data Preprocessing

#### Data Reading and Transformation

1. **Data Reading**:
   - Use the `polars` library to read the training set and misconception mapping file.
   - Convert wide-format data into long-format for easier processing.

2. **Generate Long-Format Data**:
   - Expand the text of each question's four options (A, B, C, D) to generate `QuestionId_Answer` and corresponding `AllText`.
   - `AllText` includes `ConstructName`, `SubjectName`, `QuestionText`, `CorrectAnswerText`, and `WrongAnswerText`, concatenated into a unified text field for contextual understanding by the model.
   - Map each distractor to its misconception ID and name to create long-format data.

3. **Merge Prediction Results**:
   - Read Retriever stage predictions (`oof_df.csv`) and merge predicted misconception ID lists into long-format data.

4. **Adjust Misconception ID Order**:
   - Ensure the true misconception ID is at the front of the predicted list; if not present, insert it at the top and truncate the list.

#### Data Splitting

Decide whether to use the entire dataset for training or split it into training and validation sets based on configuration.

### 2. Retriever Stage

#### Model Selection and Configuration

1. **Model Selection**:
   - Use `Qwen2.5-32B-Instruct` as the base model.

2. **LoRA Configuration**:
   - Fine-tune the model using LoRA (Low-Rank Adaptation) to reduce parameter count and training time.
   - Configure parameters such as `r=16`, `alpha=32`, `dropout=0.00`, targeting modules like `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

3. **Quantization Configuration**:
   - Use 4-bit quantization (`BitsAndBytesConfig`) to reduce memory usage and improve training efficiency.

#### Dataset and Data Loading

- Define a `QPDataset` class to input question text and corresponding candidate misconception text as queries and paragraphs.
- Use `DataLoader` for batch loading, with a custom `collate_fn` function to handle data formatting.

#### Model Training

1. **Optimizer and Scheduler**:
   - Use the `AdamW` optimizer with a learning rate of `0.0001`.
   - Adopt a `OneCycleLR` scheduler with `max_lr=0.0001`, calculating `total_steps` based on training rounds and batch size.

2. **Training Loop**:
   - Encode query and candidate misconception text for each batch, compute embeddings, and normalize them.
   - Calculate contrastive loss (`compute_no_in_batch_neg_loss`), perform backpropagation, and update gradients.

3. **Validation and Evaluation**:
   - Evaluate the model on the validation set after each training epoch, calculating MAP@25 and various Recall metrics (R@1, R@10, R@25, R@50, R@100).
   - Record and visualize training loss and learning rate curves.

4. **Model Saving**:
   - Save the fine-tuned model and tokenizer after training.

### 3. Reranker Stage

#### Model Selection and Configuration

1. **Model Selection**:
   - Use `unsloth/Qwen2.5-32B-Instruct` as the base model and FastLanguageModel for efficient inference.

2. **LoRA Configuration**:
   - Similar to the Retriever stage, fine-tune the model using LoRA, targeting the same modules.

#### Data Preprocessing

1. **Read Retriever Stage Output**:
   - Load OOF (Out-Of-Fold) predictions from the training and validation stages.

2. **Data Augmentation**:
   - Ensure the true misconception ID is within the top 5 candidates for each sample; if not, add it and shuffle the order.
   - Convert candidate misconceptions into their names and fill them into the question text to create new input formats.

3. **Template Filling**:
   - Use predefined templates to combine question text and candidate misconceptions into instruction formats for training.

#### Model Training

1. **Training Dataset**:
   - Convert preprocessed training data into Hugging Face `Dataset` format.

2. **Trainer Configuration**:
   - Use `SFTTrainer` for supervised fine-tuning, setting parameters like batch size, learning rate, optimizer type (`adamw_8bit`), weight decay, and learning rate scheduler.

3. **Training Process**:
   - Train only the response part (i.e., model-generated misconceptions) while keeping the instruction part fixed.
   - Use the `train_on_responses_only` function to optimize response generation.

4. **Model Saving**:
   - Save the fine-tuned LoRA model and tokenizer for later inference and deployment.

### 4. Evaluation and Results

#### Evaluation Metrics

- **Mean Average Precision @ 25 (MAP@25)**: Calculate the average precision of the top 25 predictions for each sample, then average across all samples.
- **Recall@K (R@K)**: Calculate the proportion of true misconceptions in the top K predictions, with common K values including 1, 10, 25, 50, and 100.

#### Retriever Results Analysis

Using the Qwen2.5-32B-Instruct model with LoRA fine-tuning, the Retriever stage achieved the following results:

- **MAP@25**: 0.4238
- **Recall@1**: 0.3017
- **Recall@10**: 0.6906
- **Recall@25**: 0.8126
- **Recall@50**: 0.8978
- **Recall@100**: 0.9391

These results indicate a high probability of including the true misconception within the top 25 predictions, with Recall@50 reaching 89.78%, demonstrating the model's effectiveness across a broader range.

#### Reranker Results Analysis

In the Reranker stage, using the unsloth/Qwen2.5-32B-Instruct model and fine-tuning with SFTTrainer, the key metrics during training were:

- **Training Loss**: 0.2672
- **Kaggle Public Leaderboard (Public LB)**: 0.54x
- **Kaggle Private Leaderboard (Private LB)**: 0.50x

The Reranker model played a critical role in improving the final MAP@25 score, with leaderboard results indicating good generalization in practical testing.
