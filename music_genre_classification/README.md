
# 🎵 Music Genre Classification
🎵 CNN-based Music Genre Classifier using Python, Librosa, and Keras


## Features
- Extract MFCC from audio
- Train a CNN to classify 10 genres
- Upload custom audio to predict genre
- Optional spectrogram visualization

## How to Run
1. Place the GTZAN dataset in `data/genres/`
2. Run `notebooks/genre_classification.ipynb` to train the model
3. Use `utils/predict_audio.py` to classify a custom audio file

## Requirements
```bash
pip install -r requirements.txt
```

## Demo
```bash
python utils/predict_audio.py
```
