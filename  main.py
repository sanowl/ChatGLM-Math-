import logging
from train import train_math_critique
from evaluate import evaluate_model
from rft import rejective_fine_tuning
from dpo import direct_preference_optimization
from config import Config
from utils import setup_logging

# Setup logging
logger = setup_logging(Config.LOGGING_DIR)

if __name__ == "__main__":
    logger.info("Starting the Self-Critique pipeline...")

    # Train the initial Math-Critique model
    logger.info("Training the Math-Critique model...")
    train_math_critique()

    # Evaluate the initial model
    logger.info("Evaluating the initial Math-Critique model...")
    evaluate_model()

    # Perform Rejective Fine-Tuning (RFT)
    logger.info("Performing Rejective Fine-Tuning (RFT)...")
    rejective_fine_tuning()

    # Perform Direct Preference Optimization (DPO)
    logger.info("Performing Direct Preference Optimization (DPO)...")
    direct_preference_optimization()

    # Final evaluation of the model
    logger.info("Final evaluation of the model...")
    evaluate_model()

    logger.info("Self-Critique pipeline completed.")
