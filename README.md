# Riichi Mahjong Discard Analyzer

A neural network that recommends the best tile to discard in Riichi Mahjong.

## How it works
- Generates 100,000 training samples using a shanten algorithm
- Trains a neural network (MahjongNet) to predict the optimal discard
- Achieves ~63% accuracy on test data

## Features
- Algorithmic hand analyzer (shanten-based)
- Neural network discard prediction
- Supports all 34 tile types (man, pin, sou, honors)

## Technologies
- Python
- PyTorch
- scikit-learn
- python-mahjong

## How to run
pip install torch scikit-learn mahjong numpy
python mahjong_stockfish.py
