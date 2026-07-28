import streamlit as st
import joblib
import zipfile
import io
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
import urllib.request # [수정 1] urllib 모듈 임포트 추가

# --- 1. 모델 및 데이터 경로/다운로드 설정 ---
DATASET_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"
MODEL_PATH = Path("lesson06_vision_model.joblib")
ZIP_PATH = Path("KolektorSDD.zip") # [수정 2] /content/ 경로 제거, 현재 폴더에 저장

# 데이터셋 파일이 없으면 다운로드 진행
if not ZIP_PATH.exists():
    with st.spinner("데이터셋(KolektorSDD.zip)을 다운로드 중입니다. 잠시만 기다려주세요..."):
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        st.success("데이터셋 다운로드 완료!")

# --- 2. 헬퍼 함수 정의 ---
FEATURE_SIZE = (64, 160)

def quality_metrics(image):
    a = np.asarray(image.resize(FEATURE_SIZE), dtype=np.float32)
    gx = np.diff(a, axis=1, prepend=a[:, :1])
    gy = np.diff(a, axis=0, prepend=a[:1, :])
    lap = (-4*a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    return {
        "brightness": float(a.mean()), 
        "contrast": float(a.std()),
        "sharpness": float(lap.var()), 
        "mean_gradient": float(np.hypot(gx,gy).mean())
    }

def extract_features(image):
    a = np.asarray(image.resize(FEATURE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)/255
    gx = np.diff(a, axis=1, prepend=a[:, :1])
    gy = np.diff(a, axis=0, prepend=a[:1, :])
    mag = np.hypot(gx, gy)
    ori = (np.degrees(np.arctan2(gy, gx))+180)%180
    hog, bins = [], np.linspace(0, 180, 10)
    for row in range(0, 160, 8):
        for col in range(0, 64, 8):
            hist, _ = np.histogram(ori[row:row+8,col:col+8], bins=bins, weights=mag[row:row+8,col:col+8])
            hog.extend(hist/(hist.sum()+1e-6))
    intensity, _ = np.histogram(a, bins=16, range=(0,1), density=True)
    percentiles = np.percentile(a, [1,5,25,50,75,95,99])
    extra = [a.mean(), a.std(), mag.mean(), np.percentile(mag,90), np.percentile(mag,99)]
    return np.concatenate([hog, intensity, percentiles, extra])

def read_single_image(archive, image_name):
    """ZIP 아카이브에서 단일 이미지를 읽습니다."""
    with archive.open(image_name) as file:
        image = Image.open(io.BytesIO(file.read())).convert("L")
    return image

# --- 3. 모델 로드 및 전역 변수 설정 ---
@st.cache_resource
def load_model_bundle():
    try:
        model_bundle = joblib.load(MODEL_PATH)
        return model_bundle
    except FileNotFoundError:
        st.error(f"모델 파일이 없습니다: {MODEL_PATH}. 깃헙 저장소에 파일이 있는지 확인해주세요.")
        st.stop()

model_bundle = load_model_bundle()
model = model_bundle["model"]
OPERATING_THRESHOLD = model_bundle["operating_threshold"]
limits = model_bundle["quality_limits"]
CLASS_NAMES = model_bundle["class_names"]

def quality_ok(q):
    return (
        limits["brightness_low"] <= q["brightness"] <= limits["brightness_high"]
        and q["contrast"] >= limits["contrast_low"]
        and q["sharpness"] >= limits["sharpness_low"]
    )

# --- 4. Streamlit UI --- 
st.set_page_config(layout="centered")
st.title("🏭 비전 품질 검사 PoC 데모")
st.markdown("Kolektor Surface-Defect Dataset의 실제 산업 이미지로 불량 여부를 판별합니다.")

# 이미지 목록 로드
@st.cache_data
def get_image_names_from_zip(zip_file_path):
    try:
        with zipfile.ZipFile(zip_file_path) as archive:
            image_names = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".jpg")
            )
        return image_names
    except FileNotFoundError:
        st.error(f"데이터 ZIP 파일이 없습니다: {zip_file_path}. 서버에서 다운로드하지 못했습니다.")
        st.stop()

image_names_list = get_image_names_from_zip(ZIP_PATH)

if not image_names_list:
    st.warning("ZIP 파일에서 이미지를 찾을 수 없습니다.")
    st.stop()

selected_image_name = st.selectbox(
    "이미지 선택:",
    image_names_list,
    index=0
)

if st.button("불량 여부 판별"): 
    if selected_image_name:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            image_to_predict = read_single_image(archive, selected_image_name)

            # 원본 이미지 표시 (작게 리사이즈)
            st.image(image_to_predict.resize((image_to_predict.width // 4, image_to_predict.height // 4)),
                     caption=f"선택된 이미지: {selected_image_name}", use_column_width=False)

            # 1. 품질 지표 계산
            q_metrics = quality_metrics(image_to_predict)
            quality_gate_passed = quality_ok(q_metrics)

            # 2. 특징 추출
            features_for_prediction = extract_features(image_to_predict).reshape(1, -1)

            # 3. 모델 예측
            defect_probability = model.predict_proba(features_for_prediction)[:, 1][0]
            prediction_label_idx = int(defect_probability >= OPERATING_THRESHOLD)
            classification_result = CLASS_NAMES[prediction_label_idx]

            # 4. 라우팅 상태 결정
            routing_status = ""
            if not quality_gate_passed:
                routing_status = "RECAPTURE_OR_HUMAN_REVIEW (이미지 품질 미달)"
            elif classification_result == "DEFECT":
                routing_status = "DEFECT_CANDIDATE_REVIEW (불량 후보)"
            else:
                routing_status = "POLICY_PASS (정상)"

            st.subheader("--- 예측 결과 ---")
            st.write(f"**불량 확률:** `{defect_probability:.4f}`")
            st.write(f"**예측:** `{classification_result}`")
            st.write(f"**품질 게이트 통과:** `{quality_gate_passed}`")
            st.write(f"**라우팅 상태:** `{routing_status}`")

            if not quality_gate_passed:
                st.warning("\n[안내] 이미지 품질이 기준 미달입니다. 재촬영 또는 수동 검토가 필요합니다.")
            elif classification_result == "DEFECT":
                st.error("\n[안내] 이 이미지는 불량 후보로 분류되었습니다. 추가 검토가 필요합니다.")
            else:
                st.success("\n[안내] 이 이미지는 정상으로 분류되었습니다.")
