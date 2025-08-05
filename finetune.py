import os
from dotenv import load_dotenv
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


# Load Configuration from .env file 
load_dotenv()

# Get paths from environment variables with sensible defaults
DATASET_PATH = os.getenv("DATASET_PATH")
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH")
PRETRAINED_MODEL_PATH = os.getenv("PRETRAINED_MODEL_PATH")

# Define model and dataset names
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"

print(f"Loading dataset from local folder: '{DATASET_PATH}'")
dataset = load_dataset(DATASET_PATH)

# Load Tokenizer 
print(f"Loading tokenizer from local: '{PRETRAINED_MODEL_PATH}'")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
                                          

# Preprocess Data 
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

print("Tokenizing dataset")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load Model 
print(f"Loading pre-trained model from local path: '{PRETRAINED_MODEL_PATH}'")
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_PATH, num_labels=2
)

# Computes and calculate metrics 
print("Metrics")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Training Arguments 
training_args = TrainingArguments(
    output_dir=FINETUNED_MODEL_PATH,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"
)

# Initialize and Run Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning")
trainer.train()

# Save the Final Model & Tokenizer 
print(f"Saving the fine-tuned model and tokenizer to '{FINETUNED_MODEL_PATH}'")
trainer.save_model(FINETUNED_MODEL_PATH)
tokenizer.save_pretrained(FINETUNED_MODEL_PATH)

print(f"Fine-tuning complete. Model saved to '{FINETUNED_MODEL_PATH}'")