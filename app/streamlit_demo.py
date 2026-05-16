"""
DeepGuard — Streamlit Demo UI
Alternative web dashboard for deepfake detection.
Run with: streamlit run app/streamlit_demo.py
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .result-card-fake {
        background: linear-gradient(135deg, #7b0d0d, #ef233c);
        border-radius: 16px; padding: 24px;
        text-align: center; color: white;
        font-size: 1.8rem; font-weight: 800;
        box-shadow: 0 8px 32px rgba(239, 35, 60, 0.4);
    }
    .result-card-real {
        background: linear-gradient(135deg, #064e3b, #10b981);
        border-radius: 16px; padding: 24px;
        text-align: center; color: white;
        font-size: 1.8rem; font-weight: 800;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px; padding: 16px;
        text-align: center;
    }
    .stProgress > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Higher threshold = stricter fake detection")
    return_heatmap = st.checkbox("Generate Grad-CAM Heatmap", value=True)
    n_frames = st.slider("Frames per Video", 4, 64, 16, 4)

    st.markdown("---")
    st.markdown("### 📡 API Status")
    try:
        resp = requests.get(f"http://localhost:8000/health", timeout=3)
        if resp.status_code == 200:
            d = resp.json()
            st.success(f"✅ API Online — {d.get('device', 'N/A').upper()}")
            st.info(f"Model loaded: {'Yes' if d.get('model_loaded') else 'No (demo mode)'}")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline\n\nRun: `make api`")

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("- [API Docs](http://localhost:8000/docs)")
    st.markdown("- [MLflow UI](http://localhost:5000)")
    st.markdown("- [GitHub](https://github.com/yourusername/deepguard)")


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ DeepGuard</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:1.1rem'>"
    "AI-powered deepfake detection for images and videos</p>",
    unsafe_allow_html=True
)
st.markdown("---")

tab_image, tab_video, tab_about = st.tabs(["🖼️ Image Detection", "🎥 Video Detection", "ℹ️ About"])

# ─── IMAGE DETECTION TAB ──────────────────────────────────────────────────────
with tab_image:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("Analyzing with AI..."):
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=95)
                    buf.seek(0)

                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/predict/image",
                            files={"file": (uploaded_file.name, buf, "image/jpeg")},
                            data={"threshold": threshold, "return_heatmap": str(return_heatmap).lower()},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.session_state["img_result"] = data
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with col_result:
        st.markdown("### Analysis Result")

        if "img_result" in st.session_state:
            data = st.session_state["img_result"]
            is_fake = data["is_fake"]
            card_class = "result-card-fake" if is_fake else "result-card-real"
            emoji = "🚨 DEEPFAKE" if is_fake else "✅ AUTHENTIC"
            st.markdown(f'<div class="{card_class}">{emoji}</div>', unsafe_allow_html=True)

            st.markdown("#### 📊 Metrics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("P(FAKE)", f"{data['probability_fake']:.1%}")
            with m2:
                st.metric("P(REAL)", f"{data['probability_real']:.1%}")
            with m3:
                st.metric("Confidence", f"{data['confidence']:.1%}")

            st.markdown("**Fake Probability**")
            st.progress(data["probability_fake"])

            st.caption(f"⏱️ Inference time: {data['processing_time_ms']:.1f}ms")

            if data.get("heatmap_base64"):
                import base64
                heatmap_bytes = base64.b64decode(data["heatmap_base64"])
                heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                st.markdown("#### 🔥 Grad-CAM Heatmap")
                st.image(heatmap_img, caption="Suspicious regions highlighted in red")
        else:
            st.info("Upload an image and click Analyze to see results.")


# ─── VIDEO DETECTION TAB ──────────────────────────────────────────────────────
with tab_video:
    col_v1, col_v2 = st.columns([1, 1], gap="large")

    with col_v1:
        st.markdown("### Upload Video")
        video_file = st.file_uploader(
            "Choose video", type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )

        if video_file:
            # Save to temp
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            st.video(tmp_path)
            analyze_vid_btn = st.button("🔍 Analyze Video", type="primary", use_container_width=True)

            if analyze_vid_btn:
                with st.spinner(f"Analyzing {n_frames} frames with AI..."):
                    try:
                        with open(tmp_path, "rb") as f:
                            resp = requests.post(
                                f"{API_BASE_URL}/predict/video",
                                files={"file": (video_file.name, f, "video/mp4")},
                                data={"threshold": threshold, "n_frames": n_frames},
                                timeout=120,
                            )
                        resp.raise_for_status()
                        st.session_state["vid_result"] = resp.json()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with col_v2:
        st.markdown("### Video Analysis Result")

        if "vid_result" in st.session_state:
            data = st.session_state["vid_result"]
            is_fake = data["is_fake"]
            card_class = "result-card-fake" if is_fake else "result-card-real"
            emoji = "🚨 DEEPFAKE VIDEO" if is_fake else "✅ AUTHENTIC VIDEO"
            st.markdown(f'<div class="{card_class}">{emoji}</div>', unsafe_allow_html=True)

            st.markdown("#### 📊 Metrics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("P(FAKE)", f"{data['probability_fake']:.1%}")
            with m2:
                st.metric("Fake Frames", f"{data['fake_frame_ratio']:.1%}")
            with m3:
                st.metric("Analyzed", f"{data['n_frames_analyzed']} frames")

            # Frame probability chart
            frame_probs = data.get("frame_probabilities", [])
            if frame_probs:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 3))
                fig.patch.set_facecolor("#1a1a2e")
                ax.set_facecolor("#16213e")
                colors = ["#ef233c" if p >= threshold else "#00b4d8" for p in frame_probs]
                ax.bar(range(len(frame_probs)), frame_probs, color=colors, alpha=0.85)
                ax.axhline(y=threshold, color="white", linestyle="--", linewidth=1.5)
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("Frame", color="white")
                ax.set_ylabel("P(FAKE)", color="white")
                ax.tick_params(colors="white")
                st.pyplot(fig)

            st.caption(f"⏱️ Total processing time: {data['processing_time_ms']:.0f}ms")
        else:
            st.info("Upload a video and click Analyze to see results.")


# ─── ABOUT TAB ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## About DeepGuard

**DeepGuard** is an AI-powered deepfake detection system developed as a graduation thesis project.

### 🏗️ Architecture
- **Backbone**: EfficientNet-B4 (pretrained on ImageNet)
- **Face Detection**: MTCNN
- **Explainability**: Grad-CAM++
- **Training Dataset**: FaceForensics++ / DFDC
- **API Framework**: FastAPI
- **Demo UI**: Gradio + Streamlit

### 📊 Performance (on FaceForensics++ test set)
| Metric | Score |
|--------|-------|
| AUC-ROC | ~0.98 |
| Accuracy | ~96% |
| F1 Score | ~0.95 |

### 🔬 Supported Deepfake Methods
- FaceSwap
- Face2Face
- DeepFakes
- NeuralTextures
- FaceShifter

### 📚 References
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC Dataset](https://ai.facebook.com/datasets/dfdc/)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
    """)
