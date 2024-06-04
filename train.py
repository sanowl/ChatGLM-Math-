import torch
from transformers import Trainer, TrainingArguments
from data import load_and_preprocess_data
from model import load_model
from config import Config

def train_model():
    encoded_dataset = load_and_preprocess_data()
    model, tokenizer = load_model()

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    train_model()