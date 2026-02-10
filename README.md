# Attention Is All You Need — Transformer from Scratch

This project implements the Transformer architecture for machine translation, inspired by the seminal paper "Attention Is All You Need" (Vaswani et al., 2017). The code is written from scratch in PyTorch and trains a bilingual translation model (English ↔ Italian) using the OPUS Books dataset.

## Features
- Pure PyTorch implementation of the Transformer model
- Custom tokenization using HuggingFace Tokenizers
- Bilingual dataset handling and preprocessing
- Teacher forcing and greedy decoding for inference
- Training, validation, and evaluation metrics (BLEU, WER, CER)
- Google Colab notebook for easy experimentation

## Project Structure
```
attention is all u need/
├── attention is all u need.pdf # Original paper
├── config.py          # Configuration file
├── dataset.py         # Bilingual dataset class
├── model.py           # Transformer model components
├── train.py           # Training script
├── train_colab.ipynb  # Colab notebook for end-to-end training
├── requirements.txt   # Project dependencies
```

## Setup
For handling large notebook files:
- Python 3.8+
- PyTorch
git lfs track "train_colab.ipynb"
- tokenizers
- tqdm
The weights folder is excluded from version control and not tracked by LFS.
- tensorboard
- torchmetrics

```bash
pip install -r requirements.txt
```
Or generate requirements.txt automatically using pipreqs:
```bash
pip install pipreqs
pipreqs . --force
```



### Git LFS
For handling large model files:
```bash
git lfs install
git lfs track "weights/*.pt"
git add .gitattributes
```

## Usage
### Training
Run the training script:
```bash
python train.py
```
Or use the Colab notebook for interactive training:
- Open `train_colab.ipynb` in Google Colab
- Run all cells

### Model Checkpoints
Model weights are saved in the `weights/` directory after each epoch. To resume training, set the `preload` parameter in the config.

### Inference
After training, use the notebook or script to generate translations for new sentences. The greedy decoding function is provided for inference.

## Configuration
Edit the `config` dictionary in `config.py` or the notebook to adjust:
- Batch size
- Number of epochs
- Learning rate
- Sequence length
- Model dimensions
- Source/target languages

## Data
The project uses the OPUS Books dataset via HuggingFace Datasets:
- English ↔ Italian translation pairs
- Tokenization is performed per language and saved as JSON files

## Model Architecture
- Input Embeddings
- Positional Encoding
- Multi-Head Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections
- Encoder & Decoder stacks

## Evaluation
Validation is performed at the end of each epoch. Metrics include:
- BLEU Score
- Word Error Rate (WER)
- Character Error Rate (CER)



---
### For questions or contributions, open an issue or pull request.
