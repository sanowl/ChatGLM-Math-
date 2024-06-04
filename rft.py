from transformers import Trainer, TrainingArguments
from model import load_saved_model, save_model
from data import load_and_preprocess_data
from config import Config

def rejective_fine_tuning():
    model, tokenizer = load_saved_model(Config.SAVE_DIR)
    encoded_dataset = load_and_preprocess_data()

    # Create a dataset of incorrect answers and correct answers
    def filter_incorrect_answers(examples):
        # Logic to filter out incorrect answers
        return examples["answer"] != examples["correct_answer"]

    incorrect_dataset = encoded_dataset.filter(filter_incorrect_answers)

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
        model=model,
        args=training_args,
        train_dataset=incorrect_dataset,
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    save_model(model, tokenizer, Config.SAVE_DIR)

if __name__ == "__main__":
    rejective_fine_tuning()
