import logging
from transformers import Trainer, TrainingArguments
from data import load_and_preprocess_data
from model import load_saved_model, save_model, MathCritiqueModel
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rejective_fine_tuning():
    dataset = load_and_preprocess_data()
    model, tokenizer = load_saved_model(Config.SAVE_DIR)
    math_critique_model = MathCritiqueModel(Config.MODEL_NAME)
    math_critique_model.model = model
    math_critique_model.tokenizer = tokenizer

    logger.info("Performing Rejective Fine-Tuning (RFT)...")
    refined_data = []
    for data in dataset["train"]:
        question, reference, answer = data['question'], data['reference'], data['answer']
        critique, score = math_critique_model.score(question, reference, answer)
        if score >= Config.SCORE_THRESHOLD:
            refined_data.append(data)

    refined_dataset = Dataset.from_dict({
        "question": [d["question"] for d in refined_data],
        "reference": [d["reference"] for d in refined_data],
        "answer": [d["answer"] for d in refined_data]
    })

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
        train_dataset=refined_dataset,
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    save_model(model, tokenizer, Config.SAVE_DIR)
    logger.info(f"Model saved to {Config.SAVE_DIR}")

if __name__ == "__main__":
    rejective_fine_tuning()

