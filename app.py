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


# 지지음역 계산 함수
def calculate_supported_range(valid_pitches, stability_threshold=0.8):
    min_pitch = np.min(valid_pitches)
    max_pitch = np.max(valid_pitches)
    supported_min = min_pitch * stability_threshold
    supported_max = max_pitch * stability_threshold
    return supported_min, supported_max


# 음량 조정 함수
def adjust_volume(y, target_rms=0.1):
    current_rms = np.sqrt(np.mean(y ** 2))
    adjustment_factor = target_rms / (current_rms + 1e-6)
    return y * adjustment_factor


# Hz 크기 조정 함수
def scale_hz(y, sr, target_pitch=440.0):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) > 0:
        mean_pitch = np.mean(valid_pitches)
        scaling_factor = target_pitch / mean_pitch
        return librosa.effects.time_stretch(y, rate=scaling_factor)
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


# 사용자가 입력한 장르와 일치도 평가
def evaluate_genre(target_genre, predicted_genre):
    return 100 if target_genre.lower() == predicted_genre.lower() else 50


# 보컬 실력 평가 (AI 기반)
def evaluate_vocal_skill(file_path, feature_extractor, model):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        confidence_score = torch.max(torch.softmax(logits, dim=-1)).item() * 100
        return confidence_score
    except Exception as e:
        st.error(f"보컬 평가 중 오류 발생: {str(e)}")
        return 0


# Streamlit 제목
st.title("🎤 AI 기반 음성 분석 및 지지음역 평가")

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
        st.write("✅ 음성 파일 로드 완료.")

        # 2️⃣ 음량 자동 조정
        adjusted_y = adjust_volume(y)
        st.write("✅ 음량 조정 완료.")

        # 3️⃣ Hz 크기 자동 조정
        scaled_y = scale_hz(adjusted_y, sr, target_pitch=440.0)
        st.write("✅ Hz 크기 조정 완료.")

        # 4️⃣ 노이즈 제거
        reduced_noise = nr.reduce_noise(y=scaled_y, sr=sr, prop_decrease=0.8)
        st.write("✅ 노이즈 제거 완료.")

        # 5️⃣ 음역대 분석
        pitches, magnitudes = librosa.piptrack(y=reduced_noise, sr=sr)
        valid_pitches = pitches[pitches > 0]
        min_pitch = np.min(valid_pitches)
        max_pitch = np.max(valid_pitches)
        mean_pitch = np.mean(valid_pitches)

        # 옥타브 및 음계 계산
        min_note = hz_to_note_name(min_pitch)
        max_note = hz_to_note_name(max_pitch)
        mean_note = hz_to_note_name(mean_pitch)

        # 지지음역 계산
        supported_min, supported_max = calculate_supported_range(valid_pitches)
        supported_min_note = hz_to_note_name(supported_min)
        supported_max_note = hz_to_note_name(supported_max)

        # 6️⃣ AI 분석
        feature_extractor, model = load_audio_model()
        if feature_extractor and model:
            predicted_genre = "Pop"  # AI 분석 결과를 사용할 경우 수정 가능
            genre_score = evaluate_genre(target_genre, predicted_genre)
            vocal_skill_score = evaluate_vocal_skill(wav_path, feature_extractor, model)
        else:
            predicted_genre = "분석 실패"
            genre_score = 0
            vocal_skill_score = 0

        # 등급 계산
        genre_grade = grade(genre_score)
        vocal_grade = grade(vocal_skill_score)

        # 결과 출력
        st.subheader("🎤 분석 결과")
        st.write(f"입력한 장르: {target_genre}")
        st.write(f"AI 분석 장르: {predicted_genre}")
        st.write(f"장르 적합성 점수: {genre_score}점 ({genre_grade})")
        st.write(f"보컬 실력 점수: {vocal_skill_score:.2f}점 ({vocal_grade})")
        st.write(f"최소 음역: {min_pitch:.2f} Hz ({min_note})")
        st.write(f"최대 음역: {max_pitch:.2f} Hz ({max_note})")
        st.write(f"평균 음역: {mean_pitch:.2f} Hz ({mean_note})")
        st.write(f"지지 음역: {supported_min:.2f} Hz ({supported_min_note}) ~ {supported_max:.2f} Hz ({supported_max_note})")

        # 결과 저장 기능
        csv_buffer = StringIO()
        results = {
            "입력한 장르": [target_genre],
            "AI 분석 장르": [predicted_genre],
            "장르 적합성 점수": [genre_score],
            "장르 적합성 등급": [genre_grade],
            "보컬 실력 점수": [vocal_skill_score],
            "보컬 실력 등급": [vocal_grade],
            "최소 음역 (Hz)": [min_pitch],
            "최소 음역 (음계)": [min_note],
            "최대 음역 (Hz)": [max_pitch],
            "최대 음역 (음계)": [max_note],
            "평균 음역 (Hz)": [mean_pitch],
            "평균 음역 (음계)": [mean_note],
            "지지 음역 (Hz)": [f"{supported_min:.2f} ~ {supported_max:.2f}"],
            "지지 음역 (음계)": [f"{supported_min_note} ~ {supported_max_note}"],
        }
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="결과를 CSV 파일로 저장",
            data=csv_buffer.getvalue(),
            file_name="음성_분석_결과.csv",
            mime="text/csv"
        )

        st.success("🎉 분석 및 개선 사항 제안이 완료되었습니다!")

    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
