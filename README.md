# Finetuning Gamma Antonyms

## Project Overview
Finetuning_Gamma_antonyms is a project designed to fine-tune the Gamma model for generating antonyms of words using causal language modeling (CLM). This project leverages Hugging Face’s transformers library, dynamic quantization, and LoRA (Low-Rank Adaptation) for model finetuning, enabling efficient model inference and training for natural language processing (NLP) tasks such as generating antonyms.

## Project Structure
```bash
Finetuning_Gamma_antonyms/
├── models/                
├── outputs/              
├── data/                 
├── inference.py 
├── dataset.py
├── evaluate.py
├── training.py 
├── main.py                
├── README.md              
└── requirements.txt
```
# Dataset

The dataset used for fine-tuning consists of a CSV file named antonyms.csv, which is located in the data/ directory. This file contains pairs of words and their antonyms in the following format:

## Key Features:

    - Model quantization using BitsAndBytes for memory-efficient inference.
    - LoRA-based finetuning to reduce computational costs.
    - Utilization of Hugging Face’s transformers library for model loading and generation.
    - Training and inference pipelines for handling the antonym generation task.

## Prerequisites

- Ensure you have the following installed:
    - Python 3.8 or higher
    - PyTorch (CUDA support optional)
    - Hugging Face transformers library
    - bitsandbytes for quantization
    - peft for LoRA configuration
    - Kaggle environment (optional) if using Kaggle for training or inference

## Installation
```bash
# Clone the repository
git clone https://github.com/Shymaa2611/Finetuning_Gamma_antonyms.git
cd Finetuning_Gamma_antonyms
```
2- Install the required dependencies
```bash
pip install -r requirements.txt

```

3- Optionally, configure your environment for efficient model loading and finetuning (e.g., CUDA for GPU-based training).

## Training Project
```bash
 !python main.py
```
## Inference
!python inference.py

## Checkpoint
- can download from : https://drive.google.com/file/d/1NThgMZQLc4hWvLbzzki7IO3W9rPY3eYF/view?usp=drive_link
## Acknowledgments
    Hugging Face for the transformers library.
    LoRA: Low-Rank Adaptation for efficient finetuning.
    BitsAndBytes for quantization support.




