
import librosa
import numpy as np
from keras.models import load_model

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

def predict(audio_path, model_path="model/cnn_model.h5"):
    model = load_model(model_path)
    y, sr = librosa.load(audio_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    max_len = 130
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    X = mfcc[np.newaxis, ..., np.newaxis]
    prediction = model.predict(X)[0]
    top_index = np.argmax(prediction)
    
    return GENRES[top_index], float(prediction[top_index])
