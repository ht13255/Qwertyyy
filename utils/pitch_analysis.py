import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def analyze_pitch_deep(y, sr, model):
    """
    딥러닝 모델을 사용해 음역대 분석.
    """
    inputs = model.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    outputs = model.model(**inputs)
    logits = outputs.logits
    predicted_ids = logits.argmax(dim=-1)
    predicted_pitch = model.processor.batch_decode(predicted_ids)

    # 정제된 결과 반환
    return {
        "min": np.min(predicted_pitch),
        "max": np.max(predicted_pitch),
        "mean": np.mean(predicted_pitch),
        "stable_pitches": predicted_pitch
    }

def analyze_supported_range(pitches):
    """
    지지 음역 계산 (Stable Pitch Range).
    """
    if len(pitches) > 0:
        min_pitch = np.percentile(pitches, 10)  # 하위 10%
        max_pitch = np.percentile(pitches, 90)  # 상위 10%
        return min_pitch, max_pitch
    return 0, 0