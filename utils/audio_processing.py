import librosa
import numpy as np
import noisereduce as nr

def preprocess_audio(uploaded_file):
    """
    음성 데이터 로드 및 전처리 (노이즈 제거 포함).
    """
    y, sr = librosa.load(uploaded_file, sr=None)
    y = librosa.util.normalize(y)
    y = nr.reduce_noise(y=y, sr=sr)
    return y, sr