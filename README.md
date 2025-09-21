# Multilingual Emotion Detection (Natural Language Processing, Deep Learning)

This project implements a multilingual multi-label emotion classifier using DistilBERT (English) and mBERT (multilingual) with LSTM+attention in PyTorch and Hugging Face Transformers. The model improves emotion detection performance across languages while handling class imbalance.

- F1-macro improved from baseline 0.372 → 0.620 (English) and 0.811 (Multilingual)
- Leverages language-aware weighting, imbalance-aware losses (Focal, Asymmetric Focal, Weighted BCE)
- Provides interpretability using LIME and attention visualization

## Features
- Multi-label emotion classification for multiple languages
- Handles imbalanced datasets effectively
- Provides interpretability insights for predictions

## Installation
```bash
pip install torch transformers scikit-learn numpy pandas matplotlib seaborn lime
