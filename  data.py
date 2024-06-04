from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

def load_and_preprocess_data():
    dataset = load_dataset("gsm8k", "main")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(examples["question"], truncation=True, padding="max_length")

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    return encoded_dataset