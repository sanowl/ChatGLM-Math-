# ChatGLM-Math

ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline.

## Description

This project implements a Self-Critique pipeline to enhance the mathematical problem-solving capabilities of Large Language Models (LLMs). It involves training a Math-Critique model, performing rejective fine-tuning (RFT), and direct preference optimization (DPO).

## Installation

```bash
pip install -r requirements.txt


Training the Model:

python train.py


Evaluating the Model:

python evaluate.py


Project Structure

	•	config.py: Configuration settings for the project.
	•	data.py: Data loading and preprocessing functions.
	•	model.py: Model definitions and loading functions.
	•	train.py: Training loop for the model.
	•	evaluate.py: Evaluation script for the model.
	•	main.py: Main script to run the complete pipeline.
	•	utils.py: Utility functions.

License

MIT License