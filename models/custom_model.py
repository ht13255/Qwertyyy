from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def load_model():
    """
    Hugging Face의 Wav2Vec2 모델 로드.
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return {"processor": processor, "model": model}