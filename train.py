import logging
import os
from transformers import Trainer, TrainingArguments
from data import load_and_preprocess_data
from model import MathCritiqueModel, save_model
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_math_critique():
    dataset = load_and_preprocess_data()
    model = MathCritiqueModel(Config.MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        logging_dir=Config.LOGGING_DIR,
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=model.tokenizer,
    )
    trainer.train()
    save_model(model.model, model.tokenizer, Config.SAVE_DIR)
    logger.info(f"Model saved to {Config.SAVE_DIR}")

if __name__ == "__main__":
    train_math_critique()
