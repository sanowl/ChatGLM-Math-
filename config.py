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
