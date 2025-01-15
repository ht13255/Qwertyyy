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


# Streamlit 제목
st.title("🎤 다양한 음성 파일 분석 및 음량 조정")

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
        st.write("✅ 음성 파일 로드 완료.")

        # 2️⃣ 음량 조정
        adjusted_y = adjust_volume(y)
        st.write("✅ 음량 조정 완료.")

        # 3️⃣ 노이즈 제거
        reduced_noise = nr.reduce_noise(y=adjusted_y, sr=sr, prop_decrease=0.8)
        st.write("✅ 노이즈 제거 완료.")

        # 4️⃣ 음역대 분석
        pitches, magnitudes = librosa.piptrack(y=reduced_noise, sr=sr)
        valid_pitches = pitches[pitches > 0]
        min_pitch = np.min(valid_pitches)
        max_pitch = np.max(valid_pitches)
        mean_pitch = np.mean(valid_pitches)

        # 옥타브 및 음계 계산
        min_note = hz_to_note_name(min_pitch)
        max_note = hz_to_note_name(max_pitch)
        mean_note = hz_to_note_name(mean_pitch)

        # 결과 출력
        st.subheader("🎤 분석 결과")
        st.write(f"최소 음역: {min_pitch:.2f} Hz ({min_note})")
        st.write(f"최대 음역: {max_pitch:.2f} Hz ({max_note})")
        st.write(f"평균 음역: {mean_pitch:.2f} Hz ({mean_note})")

        # 시각적 결과
        st.subheader("📊 시각적 결과")
        fig, ax = plt.subplots()
        librosa.display.waveshow(reduced_noise, sr=sr, ax=ax)
        ax.set(title="음량 조정 및 노이즈 제거된 음성 파형")
        st.pyplot(fig)

        st.success("🎉 분석 및 개선 사항 제안이 완료되었습니다!")

    except Exception as e:
        st.error(f"오류 발생: {str(e)}")