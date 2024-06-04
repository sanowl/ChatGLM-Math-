from datasets import load_dataset, DatasetDict

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
