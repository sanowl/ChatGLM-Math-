from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    return model, tokenizer