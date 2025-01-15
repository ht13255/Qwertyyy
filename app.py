import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import pandas as pd
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
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


# Hz를 옥타브와 음계로 변환
def hz_to_note_name(hz):
    if hz <= 0:
        return "Unknown"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_number = round(12 * np.log2(hz / 440.0) + 69)
    note = note_names[midi_number % 12]
    octave = (midi_number // 12) - 1
    return f"{octave}옥타브 {note}"


# 음량 조정 함수
def adjust_volume(y, target_rms=0.1):
    current_rms = np.sqrt(np.mean(y ** 2))
    adjustment_factor = target_rms / (current_rms + 1e-6)
    return y * adjustment_factor


# 정확한 음정 추출을 위한 YIN 기반 분석
def analyze_pitch(y, sr):
    pitches = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitches = pitches[np.isfinite(pitches)]  # NaN 제거
    return pitches


# 오디오 데이터 클리닝 함수
def clean_audio(y):
    """
    NaN 또는 Infinity 값을 제거하고 클리닝된 오디오 데이터를 반환.
    """
    if not np.all(np.isfinite(y)):  # NaN 또는 Infinity가 있는지 확인
        y = np.nan_to_num(y)  # NaN은 0으로, Infinity는 큰 유한 값으로 변환
    return y


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
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        return feature_extractor, model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        return None, None


# Streamlit 제목
st.title("🎤 AI 기반 현실적 음성 분석 및 장르 적합성 평가")

# 사용자 입력
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

        # 1️⃣ 음성 로드
        y, sr = librosa.load(wav_path, sr=None)
        y = clean_audio(y)
        st.write("✅ 음성 파일 로드 완료.")

        # 2️⃣ 음량 자동 조정
        adjusted_y = adjust_volume(y)
        st.write("✅ 음량 조정 완료.")

        # 3️⃣ 노이즈 제거
        reduced_noise = nr.reduce_noise(y=adjusted_y, sr=sr, prop_decrease=0.8)
        reduced_noise = clean_audio(reduced_noise)
        st.write("✅ 노이즈 제거 완료.")

        # 4️⃣ 정확한 음정 분석 (YIN)
        pitches = analyze_pitch(reduced_noise, sr)
        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        mean_pitch = np.mean(pitches)

        # 옥타브 및 음계 계산
        min_note = hz_to_note_name(min_pitch)
        max_note = hz_to_note_name(max_pitch)
        mean_note = hz_to_note_name(mean_pitch)

        # 5️⃣ 분석 결과 출력
        st.subheader("🎤 분석 결과")
        st.write(f"최소 음역: {min_pitch:.2f} Hz ({min_note})")
        st.write(f"최대 음역: {max_pitch:.2f} Hz ({max_note})")
        st.write(f"평균 음역: {mean_pitch:.2f} Hz ({mean_note})")

        # 6️⃣ 결과 저장
        results = {
            "최소 음역 (Hz)": [min_pitch],
            "최대 음역 (Hz)": [max_pitch],
            "평균 음역 (Hz)": [mean_pitch],
            "최소 음역 (음계)": [min_note],
            "최대 음역 (음계)": [max_note],
            "평균 음역 (음계)": [mean_note],
        }
        csv_buffer = StringIO()
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

        st.download_button(
            label="결과를 CSV 파일로 저장",
            data=csv_buffer.getvalue(),
            file_name="음성_분석_결과.csv",
            mime="text/csv"
        )
        st.success("🎉 분석이 완료되었습니다!")

    except Exception as e:
        st.error(f"오류 발생: {str(e)}")