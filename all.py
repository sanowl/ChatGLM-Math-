import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class to manage all settings
class Config:
    MODEL_NAME = "gpt2"  # Change to a valid model identifier
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    OUTPUT_DIR = "./results"
    LOGGING_DIR = "./logs"
    SAVE_DIR = "./model"
    SCORE_THRESHOLD = 7

# MathCritiqueModel class to evaluate and score answers
class MathCritiqueModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def score(self, question, reference, answer):
        critique = self.evaluate_answer(question, reference, answer)
        score = self.calculate_score(critique)
        return critique, score

    def evaluate_answer(self, question, reference, answer):
        # Tokenize the input question
        inputs = self.tokenizer(question, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Evaluate the generated answer against the reference
        critique = {
            "question": question, 
            "reference": reference, 
            "answer": answer, 
            "generated": generated_text
        }
        return critique

    def calculate_score(self, critique):
        # Real scoring logic: use a more advanced method to compare answers
        reference = critique["reference"]
        generated = critique["generated"]

        # Example scoring logic (to be replaced with a more sophisticated method)
        score = 10 if generated.strip() == reference.strip() else 0
        return score

# SelfCritiquePipeline class to manage the entire pipeline
class SelfCritiquePipeline:
    def __init__(self, model_name):
        self.math_critique_model = MathCritiqueModel(model_name)

    def train_math_critique(self, dataset):
        logger.info("Training Math-Critique model...")
        # Implement the actual training logic
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
            model=self.math_critique_model.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.math_critique_model.tokenizer,
        )
        trainer.train()
        self.save_model()

    def save_model(self):
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        self.math_critique_model.model.save_pretrained(Config.SAVE_DIR)
        self.math_critique_model.tokenizer.save_pretrained(Config.SAVE_DIR)
        logger.info(f"Model saved to {Config.SAVE_DIR}")

    def load_model(self):
        self.math_critique_model.model = AutoModelForCausalLM.from_pretrained(Config.SAVE_DIR)
        self.math_critique_model.tokenizer = AutoTokenizer.from_pretrained(Config.SAVE_DIR)
        logger.info(f"Model loaded from {Config.SAVE_DIR}")

    def rejective_fine_tuning(self, dataset):
        logger.info("Performing Rejective Fine-Tuning (RFT)...")
        refined_data = []
        for data in dataset["train"]:
            question, reference, answer = data['question'], data['reference'], data['answer']
            critique, score = self.math_critique_model.score(question, reference, answer)
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
            model=self.math_critique_model.model,
            args=training_args,
            train_dataset=refined_dataset,
            eval_dataset=dataset["test"],
            tokenizer=self.math_critique_model.tokenizer,
        )
        trainer.train()
        self.save_model()

    def direct_preference_optimization(self, dataset):
        logger.info("Performing Direct Preference Optimization (DPO)...")
        pairs = []
        for data in dataset["train"]:
            question, reference, answer = data['question'], data['reference'], data['answer']
            critique, score = self.math_critique_model.score(question, reference, answer)
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
            model=self.math_critique_model.model,
            args=training_args,
            train_dataset=pairs_dataset,
            eval_dataset=dataset["test"],
            tokenizer=self.math_critique_model.tokenizer,
        )
        trainer.train()
        self.save_model()

# Load and preprocess the dataset
def load_and_preprocess_data():
    dataset = load_dataset("gsm8k", "main")
    def preprocess_function(examples):
        return {
            "question": examples["question"],
            "reference": examples["answer"],
            "answer": examples["answer"]
        }
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    return DatasetDict({"train": encoded_dataset["train"], "test": encoded_dataset["test"]})

if __name__ == "__main__":
    model_name = Config.MODEL_NAME
    pipeline = SelfCritiquePipeline(model_name)
    dataset = load_and_preprocess_data()

    # Train the initial Math-Critique model
    pipeline.train_math_critique(dataset)

    # Perform Rejective Fine-Tuning (RFT)
    pipeline.rejective_fine_tuning(dataset)

    # Perform Direct Preference Optimization (DPO)
    pipeline.direct_preference_optimization(dataset)
