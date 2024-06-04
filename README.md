# ChatGLM-Math

ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline.

## Description

This project implements a Self-Critique pipeline to enhance the mathematical problem-solving capabilities of Large Language Models (LLMs). It involves training a Math-Critique model, performing rejective fine-tuning (RFT), and direct preference optimization (DPO).

## Installation

1. Clone the repository:
    ```bash
  [  [](https://github.com/sanowl/ChatGLM-Math-.git)](https://github.com/sanowl/ChatGLM-Math-.git)
    cd ChatGLM-Math
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Training the Math-Critique Model:**
    ```bash
    python train.py
    ```

2. **Evaluating the Model:**
    ```bash
    python evaluate.py
    ```

3. **Performing Rejective Fine-Tuning (RFT):**
    ```bash
    python rft.py
    ```

4. **Performing Direct Preference Optimization (DPO):**
    ```bash
    python dpo.py
    ```

5. **Running the Complete Pipeline:**
    ```bash
    python main.py
    ```

## Project Structure

- `config.py`: Configuration settings for the project.
- `data.py`: Data loading and preprocessing functions.
- `model.py`: Model definitions and utility functions.
- `train.py`: Training script for the Math-Critique model.
- `evaluate.py`: Evaluation script for the model.
- `rft.py`: Script for performing Rejective Fine-Tuning (RFT).
- `dpo.py`: Script for performing Direct Preference Optimization (DPO).
- `main.py`: Main script to run the complete pipeline.
- `utils.py`: Utility functions (currently empty, can be expanded as needed).

## License

MIT License
