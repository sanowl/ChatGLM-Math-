import logging
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from data import load_and_preprocess_data
from model import load_saved_model, save_model, MathCritiqueModel
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def direct_preference_optimization():
    dataset = load_and_preprocess_data()
    model, tokenizer = load_saved_model(Config.SAVE_DIR)
    math_critique_model = MathCritiqueModel(Config.MODEL_NAME)
    math_critique_model.model = model
    math_critique_model.tokenizer = tokenizer

    logger.info("Performing Direct Preference Optimization (DPO)...")
    pairs = []
    for data in dataset["train"]:
        question, reference, answer = data['question'], data['reference'], data['answer']
        critique, score = math_critique_model.score(question, reference, answer)
        if score >= Config.SCORE_THRESHOLD:
            pairs.append({"question": question, "better_answer": answer, "worse_answer": reference})

    pairs_dataset = Dataset.from_dict({
        "question": [p["question"] for p in pairs],
        "better_answer": [p["better_answer"] for p in pairs],
        "worse_answer": [p["worse_answer"] for p in pairs]
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
        train_dataset=pairs_dataset,
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    save_model(model, tokenizer, Config.SAVE_DIR)
    logger.info(f"Model saved to {Config.SAVE_DIR}")

if __name__ == "__main__":
    direct_preference_optimization()
