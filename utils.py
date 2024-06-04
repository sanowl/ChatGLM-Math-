import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# Save model and tokenizer
def save_model(model, tokenizer, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

# Load model and tokenizer
def load_model(save_directory):
    model = AutoModelForCausalLM.from_pretrained(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    return model, tokenizer

# Calculate score based on reference and generated answers
def calculate_score(reference, generated):
    reference = reference.strip()
    generated = generated.strip()

    # Example scoring logic (customize as needed)
    score = 10 if reference == generated else 0
    return score
