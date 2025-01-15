import streamlit as st
from utils.audio_processing import preprocess_audio
from utils.pitch_analysis import analyze_pitch_deep, analyze_supported_range
from models.custom_model import load_model

# Streamlit 제목
st.title("🎤 딥러닝 기반 고도화된 음역대 분석")

# 사용자 입력
target_genre = st.text_input("분석할 노래 장르를 입력하세요 (예: Pop, Jazz, Rock)", value="Pop")

# 음성 파일 업로드
uploaded_file = st.file_uploader("음성을 업로드하세요 (MP3, AAC, OGG, WAV 등 지원)", type=["mp3", "wav", "ogg", "aac", "wma"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.write("업로드된 음성을 처리 중입니다...")

    try:
        # 1️⃣ 음성 데이터 전처리
        y, sr = preprocess_audio(uploaded_file)

        # 2️⃣ 딥러닝 모델 로드 및 예측
        model = load_model()
        pitch_results = analyze_pitch_deep(y, sr, model)

        # 3️⃣ 음역대 및 지지 음역 계산
        min_pitch, max_pitch, mean_pitch = pitch_results["min"], pitch_results["max"], pitch_results["mean"]
        supported_min, supported_max = analyze_supported_range(pitch_results["stable_pitches"])

        # 음계 변환
        min_note = pitch_results["min_note"]
        max_note = pitch_results["max_note"]
        mean_note = pitch_results["mean_note"]
        supported_min_note = pitch_results["supported_min_note"]
        supported_max_note = pitch_results["supported_max_note"]

        # 결과 출력
        st.subheader("🎤 분석 결과")
        st.write(f"최소 음역: {min_pitch:.2f} Hz ({min_note})")
        st.write(f"최대 음역: {max_pitch:.2f} Hz ({max_note})")
        st.write(f"평균 음역: {mean_pitch:.2f} Hz ({mean_note})")
        st.write(f"지지 음역: {supported_min:.2f} Hz ({supported_min_note}) ~ {supported_max:.2f} Hz ({supported_max_note})")

        # 결과 저장
        st.download_button("CSV 파일로 저장", data="...")

    except Exception as e:
        st.error(f"오류 발생: {str(e)}")