# Causal Language Modeling for Code Generation

This repository contains a **GPT-2-based causal language model** trained on the **CodeSearchNet** Python dataset. The model is designed to generate Python code given a natural language docstring as input.

## 📌 Features

- **Custom Tokenizer**: Built from scratch instead of using the default GPT-2 tokenizer.
- **Fine-tuned GPT-2**: Trained on Python-specific code snippets.
- **Docstring-to-Code Generation**: Generates function implementations from textual descriptions.
- **Efficient Training Pipeline**: Implements dataset preprocessing, model training, and evaluation using PyTorch and Hugging Face’s Transformers.

---

## 📂 Project Structure
causal-language-modeling/  
├── model_config.py  
├── train.py  
├── tokenizer.py  
├── preprocessor.py  
├── data_loader.py  
│── notebooks/  
│── tokenizer/  
|   |── custom_tokenizer.json  
│── README.md  

## Training Data
The model is trained on the Python subset of CodeSearchNet, a dataset containing 800K of function-docstring pairs.  

## 🛠 Technologies Used
	•	PyTorch  
	•	Transformers (Hugging Face)  
	•	Tokenizers  
	•	Datasets (Hugging Face)  

## 📜 License

This project is licensed under the MIT License. See LICENSE for details.
