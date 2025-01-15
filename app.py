import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import pandas as pd
from transformers import AutoModelForAudioClassification, AutoProcessor
import subprocess
import os
import imageio_ffmpeg as ffmpeg
import torch
import torchaudio
from io import StringIO


# 평가 점수 계산 함수
def grade(score):
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 75:
        return "C+"
    elif score >= 70:
        return "C"
    elif score >= 65:
        return "D+"
    elif score >= 60:
        return "D"
    elif score >= 50:
        return "F+"
    else:
        return "F-"


# 업로드된 음성을 WAV 형식으로 변환
def convert_to_wav(uploaded_file, output_path="temp_audio.wav"):
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    with open("temp_input_file", "wb") as f:
        f.write(uploaded_file.read())
    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", "temp_input_file", output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_path
    except Exception as e:
        st.error(f"파일 변환 중 오류 발생: {str(e)}")
        return None


# Hugging Face 모델 및 처리기 로드
def load_audio_model():
    model_name = "superb/hubert-large-superb-er"
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        return None, None


# 예측 함수
def classify_audio(file_path, processor, model):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        return model.config.id2label[predicted_id]
    except Exception as e:
        st.error(f"오디오 분류 중 오류 발생: {str(e)}")
        return None


# Streamlit 제목
st.title("🎤 다양한 음성 파일 분석 및 지지음역 계산 애플리케이션")

# 사용자 입력 장르
target_genre = st.text_input("분석할 노래 장르를 입력하세요 (예: Pop, Jazz, Rock)", value="Pop")

# 음성 파일 업로드
uploaded_file = st.file_uploader("음성을 업로드하세요 (MP3, AAC, OGG, WAV 등 지원)", type=["mp3", "wav", "ogg", "aac", "wma"])

if uploaded_file:
    try:
        st.audio(uploaded_file, format="audio/wav")
        st.write("업로드된 음성을 처리 중입니다...")

        # 파일 변환
        wav_path = convert_to_wav(uploaded_file)
        if not wav_path:
            st.error("파일 변환 실패로 분석을 중단합니다.")
            st.stop()

        # 1️⃣ 음성 로드 및 노이즈 제거
        y, sr = librosa.load(wav_path, sr=None)
        reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        st.write("✅ 노이즈 제거 완료.")

        # 2️⃣ 음성 분석
        st.write("2️⃣ 음성 분석 중...")

        # 음역대 계산
        pitches, magnitudes = librosa.piptrack(y=reduced_noise, sr=sr)
        valid_pitches = pitches[pitches > 0]
        min_pitch = np.min(valid_pitches)
        max_pitch = np.max(valid_pitches)
        mean_pitch = np.mean(valid_pitches)

        # 옥타브 계산
        min_octave = int(np.log2(min_pitch / 16.35))
        max_octave = int(np.log2(max_pitch / 16.35))

        # 지지음역 계산
        stability_threshold = 0.8
        supported_range = (min_pitch * stability_threshold, max_pitch * stability_threshold)

        # 성량 분석
        rms = librosa.feature.rms(y=reduced_noise).mean() * 100
        volume_score = min(100, max(50, rms))

        # 리듬 분석
        onset_env = librosa.onset.onset_strength(y=reduced_noise, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=reduced_noise, sr=sr, onset_envelope=onset_env)
        rhythm_accuracy = min(100, max(50, 120 / tempo * 100))

        # Hugging Face 모델 로드 및 예측
        processor, model = load_audio_model()
        if processor and model:
            genre_label = classify_audio(wav_path, processor, model)
        else:
            genre_label = "분석 실패"

        st.write("✅ 음성 분석 완료.")

        # 분석 결과 데이터프레임 생성
        results = {
            "평균 음역대 (Hz)": [mean_pitch],
            "최소 음역대 (Hz)": [min_pitch],
            "최대 음역대 (Hz)": [max_pitch],
            "지지음역 (Hz)": [f"{supported_range[0]:.2f} ~ {supported_range[1]:.2f}"],
            "옥타브 범위": [f"{min_octave} ~ {max_octave}"],
            "성량 점수": [f"{volume_score:.2f}점 ({grade(volume_score)})"],
            "리듬 정확도": [f"{rhythm_accuracy:.2f}%"],
            "AI 분석 장르": [genre_label],
            "장르 일치 여부": ["일치" if genre_label.lower() == target_genre.lower() else "불일치"],
        }
        df_results = pd.DataFrame(results)

        # 3️⃣ 결과 출력
        st.subheader("🎤 세부 분석 결과")
        st.dataframe(df_results)

        # 결과 저장 기능
        st.subheader("📂 분석 결과 저장")
        csv_buffer = StringIO()
        df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="결과를 CSV 파일로 저장",
            data=csv_buffer.getvalue(),
            file_name="음성_분석_결과.csv",
            mime="text/csv"
        )
        txt_buffer = StringIO()
        for key, value in results.items():
            txt_buffer.write(f"{key}: {value[0]}\n")
        st.download_button(
            label="결과를 TXT 파일로 저장",
            data=txt_buffer.getvalue(),
            file_name="음성_분석_결과.txt",
            mime="text/plain"
        )

        # 4️⃣ 시각적 결과
        st.subheader("📊 시각적 결과")
        fig, ax = plt.subplots()
        librosa.display.waveshow(reduced_noise, sr=sr, ax=ax)
        ax.set(title="노이즈 제거된 음성 파형")
        st.pyplot(fig)

        st.success("🎉 분석 및 개선 사항 제안이 완료되었습니다!")

        # 임시 파일 제거
        os.remove("temp_input_file")
        os.remove(wav_path)
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")