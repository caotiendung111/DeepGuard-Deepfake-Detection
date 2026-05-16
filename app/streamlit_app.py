"""
DeepGuard Streamlit UI
Giao diện Web tương tác cho hệ thống phát hiện Deepfake.
"""
import os
import subprocess
import sys
import time
import json
import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from PIL import Image

# ─── Cấu Hình Trang ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard - Phát hiện Deepfake",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS để làm đẹp giao diện theo phong cách Hyper-Precision Glass
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460) !important;
        color: #e2e8f0;
    }
    
    /* Glassmorphism panels */
    .st-emotion-cache-1y4p8pa {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Image rounding & shadow */
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Custom Result Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    .card-fake {
        border: 2px solid rgba(239, 35, 60, 0.5);
        box-shadow: 0 0 20px rgba(239, 35, 60, 0.3);
        background: linear-gradient(135deg, rgba(239, 35, 60, 0.1), rgba(0,0,0,0));
    }
    .card-fake:hover {
        box-shadow: 0 0 30px rgba(239, 35, 60, 0.5);
    }
    
    .card-real {
        border: 2px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(0,0,0,0));
    }
    .card-real:hover {
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
    }
    
    .pulse-text-fake {
        color: #ef233c;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        text-shadow: 0 0 10px rgba(239,35,60,0.5);
        animation: pulse 2s infinite;
    }
    
    .pulse-text-real {
        color: #10b981;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        text-shadow: 0 0 10px rgba(16,185,129,0.5);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Metrics customization */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.1);
        border-bottom: 2px solid #4facfe !important;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Constants & API Setup ──────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
AUTO_START_API = os.environ.get("AUTO_START_API", "1").lower() in {"1", "true", "yes", "on"}
PASTE_IMAGE_COMPONENT = components.declare_component(
    "paste_image",
    path=str(Path(__file__).parent / "paste_component"),
)

def _request_api_health(timeout: float = 2.0):
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
        if res.status_code == 200:
            return res.json()
    except requests.RequestException:
        pass
    return None


def _is_local_api_url() -> bool:
    parsed = urlparse(API_BASE_URL)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _api_port() -> str:
    parsed = urlparse(API_BASE_URL)
    return str(parsed.port or 8000)


def _start_api_server() -> None:
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "api_autostart.log", "a", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        _api_port(),
    ]

    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )


@st.cache_data(ttl=30, show_spinner=False)
def check_api_health():
    """Cache health check for 30 seconds to avoid blocking the UI on every interaction."""
    status = _request_api_health(timeout=1.5)
    if status or not AUTO_START_API or not _is_local_api_url():
        return status

    # Only attempt auto-start if we are sure we are local and not already starting
    now = time.time()
    last_start = st.session_state.get("api_autostart_time", 0)
    if now - last_start > 60: # Cooldown for autostart
        st.session_state["api_autostart_time"] = now
        try:
            _start_api_server()
        except Exception as e:
            st.session_state["api_autostart_error"] = str(e)
            return None

    # Wait a bit for the server to spin up
    for _ in range(5):
        time.sleep(1)
        status = _request_api_health(timeout=1.0)
        if status:
            return status

    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6213/6213731.png", width=100) # Placeholder logo
st.sidebar.title("🛡️ DeepGuard")
st.sidebar.markdown("**Hệ thống phát hiện Deepfake**")
st.sidebar.divider()

api_status = check_api_health()
default_threshold = 0.68
if api_status and api_status.get("threshold") is not None:
    default_threshold = round(float(api_status["threshold"]), 2)

# Cài đặt Ngưỡng
st.sidebar.subheader("⚙️ Cài đặt")
threshold = st.sidebar.slider(
    "Ngưỡng cảnh báo (Threshold)",
    min_value=0.1, max_value=0.99, value=default_threshold, step=0.01,
    help="Độ tin cậy tối thiểu để kết luận một bức ảnh/video là FAKE."
)

st.sidebar.divider()
st.sidebar.subheader("📡 Trạng thái API")
if api_status:
    tta_text = "TTA" if api_status.get("inference_tta") else "single"
    st.sidebar.success(f"🟢 Online (Model: {api_status.get('model')} | {tta_text})")
else:
    st.sidebar.error("🔴 Offline (Không thể kết nối Backend)")
    if AUTO_START_API and _is_local_api_url():
        if st.session_state.get("api_autostart_error"):
            st.sidebar.caption(f"Auto-start lỗi: {st.session_state['api_autostart_error']}")
        else:
            st.sidebar.caption("Đã thử tự khởi động API. Xem logs/api_autostart.log nếu vẫn offline.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 DeepGuard Project")


# ─── Main Content ─────────────────────────────────────────────────────────
st.title("🛡️ DeepGuard: Nhận diện Ảnh & Video Giả mạo")
st.markdown("Hệ thống tự động phân tích khuôn mặt và phát hiện các dấu hiệu can thiệp bằng Deep Learning.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Phân tích Ảnh Tĩnh", 
    "🎥 Phân tích Video", 
    "📂 Xử lý Lô (Batch)", 
    "🧠 Hiểu Mô Hình (XAI)",
    "🧪 Tự kiểm thử"
])

# ==========================================
# TAB 1: ẢNH TĨNH
# ==========================================
with tab1:
    st.header("📸 Phân tích Ảnh tĩnh")
    st.markdown("Tải lên một bức ảnh (.jpg, .png) có chứa khuôn mặt để hệ thống phân tích.")
    
    col_upload, col_result = st.columns([1, 1.2])
    
    with col_upload:
        uploaded_image = st.file_uploader("Chọn ảnh cần kiểm tra", type=["jpg", "jpeg", "png", "webp"])

        st.markdown("**Dán ảnh từ clipboard**")
        st.caption("Click vào khung bên dưới rồi nhấn Ctrl+V. Nếu không hiện, restart Streamlit để load component mới.")
        pasted_image = PASTE_IMAGE_COMPONENT(key="paste_image_component", default=None)
        pasted_data_url = st.text_area(
            "Fallback: dán URL ảnh hoặc data URL/base64 ảnh vào đây",
            height=80,
            placeholder="https://.../image.jpg hoặc data:image/png;base64,...",
        )
        image_bytes = None
        image_name = None
        image_type = "image/png"

        if uploaded_image is not None:
            image_bytes = uploaded_image.getvalue()
            image_name = uploaded_image.name
            image_type = uploaded_image.type or "image/jpeg"
        elif pasted_image and pasted_image.get("data_url"):
            try:
                _, encoded = pasted_image["data_url"].split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image_name = pasted_image.get("name") or "clipboard_image.png"
                image_type = pasted_image.get("type") or "image/png"
            except Exception as e:
                st.error(f"Không đọc được ảnh từ clipboard: {e}")
        elif pasted_data_url.strip():
            try:
                pasted_text = pasted_data_url.strip()
                if pasted_text.startswith(("http://", "https://")):
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                    response = requests.get(pasted_text, headers=headers, timeout=20)
                    response.raise_for_status()
                    image_bytes = response.content
                    image_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
                    image_name = Path(pasted_text.split("?", 1)[0]).name or "image_from_url.jpg"
                elif "," in pasted_text:
                    header, encoded = pasted_text.split(",", 1)
                    if "image/jpeg" in header:
                        image_type = "image/jpeg"
                    elif "image/webp" in header:
                        image_type = "image/webp"
                else:
                    encoded = pasted_text
                    image_bytes = base64.b64decode(encoded)
                    image_name = "pasted_base64_image.png"
            except Exception as e:
                st.error(f"Không đọc được ảnh từ ô fallback: {e}")

        if image_bytes:
            try:
                image = Image.open(BytesIO(image_bytes))
                image.verify()
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                st.error(f"Dữ liệu nhận được không phải ảnh hợp lệ: {e}")
                image = None

        if image_bytes and image is not None:
            if st.button("Phân tích ảnh này 🔍", type="primary", width='stretch'):
                if not api_status:
                    st.error("API hiện không hoạt động!")
                else:
                    with st.spinner("Đang quét khuôn mặt và phân tích..."):
                        # Gọi API
                        files = {"file": (image_name, image_bytes, image_type)}
                        try:
                            t0 = time.time()
                            response = requests.post(
                                f"{API_BASE_URL}/predict/image",
                                files=files,
                                data={"threshold": str(threshold), "return_heatmap": "true"},
                                timeout=30,
                            )
                            t1 = time.time()
                            
                            if response.status_code == 200:
                                st.session_state["img_result"] = response.json()
                                st.session_state["img_time"] = t1 - t0
                                st.session_state["img_source_name"] = image_name
                            else:
                                st.error(f"Lỗi từ server: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Không thể kết nối: {e}")

            st.image(image, caption="Ảnh gốc", width='stretch')

    with col_result:
        if "img_result" in st.session_state:
            res = st.session_state["img_result"]
            prob = res.get('probability_fake', res.get('confidence', 0.0))
            is_fake = res.get('is_fake', prob >= threshold)
            decision_confidence = res.get('confidence', max(prob, 1 - prob))
            label_text = "GIẢ MẠO (FAKE)" if is_fake else "ẢNH THẬT (REAL)"
            color = "red" if is_fake else "green"
            
            st.markdown("### Kết quả phân tích")
            
            # Khung thông báo chính (Glassmorphism)
            card_class = "card-fake" if is_fake else "card-real"
            text_class = "pulse-text-fake" if is_fake else "pulse-text-real"
            st.markdown(
                f"""
                <div class="glass-card {card_class}">
                    <h2 class="{text_class}">{label_text}</h2>
                    <p style="color: #cbd5e1; font-size: 1.2rem; margin-top: 12px; font-weight: 300;">
                        Mức độ tin cậy: <strong style="color: white; font-weight: 600;">{decision_confidence*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            
            st.caption(f"⏱️ Thời gian xử lý: {res['processing_time_ms']:.1f} ms")
            if res.get("analysis_note"):
                st.caption(f"Chẩn đoán: {res['analysis_note']}")
            if res.get("face_probability_fake") is not None and res.get("full_probability_fake") is not None:
                st.caption(
                    f"Score face/full: {res['face_probability_fake']*100:.1f}% / "
                    f"{res['full_probability_fake']*100:.1f}%"
                )
            
            # Thanh Progress bar
            st.progress(prob, text="Thang đo FAKE (0 = Thật, 1 = Giả)")
            
            # Hiển thị Heatmap
            if is_fake and res.get('heatmap_base64'):
                st.markdown("#### Bằng chứng (Grad-CAM Heatmap)")
                st.markdown("*Vùng màu đỏ/vàng là nơi AI cho rằng có dấu hiệu can thiệp.*")
                
                img_data = base64.b64decode(res['heatmap_base64'])
                heatmap_img = Image.open(BytesIO(img_data))
                st.image(heatmap_img, caption="Original | Heatmap | Overlay", width='stretch')
                
                # Nút tải báo cáo
                st.download_button(
                    label="⬇️ Tải JSON Kết quả",
                    data=json.dumps(res, indent=2),
                    file_name=f"deepguard_report_{st.session_state.get('img_source_name', 'image')}.json",
                    mime="application/json"
                )

# ==========================================
# TAB 2: VIDEO
# ==========================================
with tab2:
    st.header("🎥 Phân tích Video (Async)")
    st.markdown("Tải lên video (.mp4) tối đa 100MB. Hệ thống sẽ bóc tách khung hình và phân tích sự giả mạo theo thời gian.")
    
    vid_col1, vid_col2 = st.columns([1, 2])
    
    with vid_col1:
        uploaded_video = st.file_uploader("Chọn video cần kiểm tra", type=["mp4", "avi"])
        if uploaded_video:
            st.video(uploaded_video)
            
            if st.button("Bắt đầu quét Video 🕵️‍♂️", type="primary", width='stretch'):
                if not api_status:
                    st.error("API hiện không hoạt động!")
                else:
                    with st.spinner("Đang tải video lên server..."):
                        files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                        response = requests.post(
                            f"{API_BASE_URL}/predict/video",
                            files=files,
                            data={"threshold": threshold},
                        )
                        
                        if response.status_code == 200:
                            job_id = response.json().get("job_id")
                            st.session_state["current_job_id"] = job_id
                            st.success("Tải lên thành công! Bắt đầu phân tích nền...")
                        else:
                            st.error(f"Lỗi tải lên: {response.json().get('detail')}")
                            
    with vid_col2:
        if "current_job_id" in st.session_state:
            job_id = st.session_state["current_job_id"]
            
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Polling loop
            while True:
                res = requests.get(f"{API_BASE_URL}/predict/video/{job_id}").json()
                status = res.get("status")
                
                if status == "processing" or status == "pending":
                    status_placeholder.info("🔄 Đang xử lý... Vui lòng đợi. Cứ mỗi giây sẽ quét 1 khung hình.")
                    time.sleep(2)
                elif status == "failed":
                    status_placeholder.error(f"❌ Phân tích thất bại: {res.get('error')}")
                    break
                elif status == "done":
                    status_placeholder.success("✅ Phân tích hoàn tất!")
                    
                    # Hiện kết quả tổng quan
                    vid_prob = res.get('probability_fake', res.get('confidence', 0.0))
                    vid_fake = res.get('is_fake', vid_prob >= threshold)
                    vid_confidence = res.get('confidence', max(vid_prob, 1 - vid_prob))
                    color = "red" if vid_fake else "green"
                    
                    # Kết luận Video (Glassmorphism)
                    card_class = "card-fake" if vid_fake else "card-real"
                    text_class = "pulse-text-fake" if vid_fake else "pulse-text-real"
                    label_text = "VIDEO GIẢ MẠO" if vid_fake else "VIDEO THẬT"
                    
                    st.markdown(
                        f"""
                        <div class="glass-card {card_class}" style="margin-bottom: 24px;">
                            <h2 class="{text_class}">{label_text}</h2>
                            <p style="color: #cbd5e1; font-size: 1.1rem; margin-top: 12px; font-weight: 300;">
                                Xác suất Fake tổng thể: <strong style="color: white;">{vid_prob*100:.1f}%</strong><br>
                                Đã phân tích <b>{res['frames_analyzed']}</b> khung hình (⏱️ {res['processing_time_ms']/1000:.1f} giây).
                            </p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    
                    # Vẽ biểu đồ Plotly
                    timeline = res.get("timeline", [])
                    if timeline:
                        frames = list(range(len(timeline)))
                        df_plot = pd.DataFrame({'Frame': frames, 'Confidence': timeline})
                        
                        fig = px.area(
                            df_plot, x="Frame", y="Confidence", 
                            title="Diễn biến Xác suất FAKE theo thời gian",
                            labels={"Confidence": "Xác suất FAKE", "Frame": "Khung hình (s)"},
                            template="plotly_dark",
                            color_discrete_sequence=["#f43f5e"]
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                            font=dict(family="Outfit", color="#e2e8f0")
                        )
                        
                        # Đường Threshold
                        fig.add_hline(
                            y=threshold, line_dash="dash", line_color="orange", 
                            annotation_text="Ngưỡng cảnh báo"
                        )
                        
                        # Đánh dấu đỉnh cao nhất
                        max_idx = np.argmax(timeline)
                        max_val = timeline[max_idx]
                        
                        fig.add_trace(go.Scatter(
                            x=[max_idx], y=[max_val],
                            mode='markers',
                            marker=dict(color='red', size=12, symbol='star'),
                            name='Điểm đáng ngờ nhất',
                            hovertemplate='Frame: %{x}<br>Conf: %{y:.3f}'
                        ))
                        
                        fig.update_layout(yaxis_range=[-0.05, 1.05])
                        st.plotly_chart(fig, width='stretch')
                    
                    break


# ==========================================
# TAB 3: BATCH PROCESSING
# ==========================================
with tab3:
    st.header("📂 Xử lý lô (Nhiều ảnh)")
    st.markdown("Tiết kiệm thời gian bằng cách tải lên nhiều ảnh cùng lúc để quét hàng loạt.")
    
    uploaded_files = st.file_uploader("Chọn nhiều ảnh", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Bắt đầu quét toàn bộ", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Đang xử lý {i+1}/{total}: {file.name}")
                files = {"file": (file.name, file.getvalue(), file.type)}
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/predict/image",
                        files=files,
                        data={"threshold": threshold, "return_heatmap": "false"},
                    ).json()
                    prob = res.get('probability_fake', res.get('confidence', 0.0))
                    result_label = res.get("label", "FAKE" if prob >= threshold else "REAL")
                    results.append({
                        "Tên file": file.name,
                        "Kết quả": result_label,
                        "Xác suất FAKE": f"{prob*100:.2f}%",
                        "Tin cậy": f"{res.get('confidence', max(prob, 1 - prob))*100:.2f}%",
                        "Có mặt người": "Có" if res.get('face_detected') else "Không",
                        "Tốc độ (ms)": f"{res.get('processing_time_ms', 0):.1f}"
                    })
                except Exception as e:
                    results.append({
                        "Tên file": file.name,
                        "Kết quả": "LỖI",
                        "Xác suất FAKE": "N/A",
                        "Có mặt người": "N/A",
                        "Tốc độ (ms)": "N/A"
                    })
                
                progress_bar.progress((i + 1) / total)
                
            status_text.success("Đã hoàn tất xử lý lô!")
            
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, width='stretch')
            
            # Xuất CSV
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Tải xuống báo cáo CSV",
                csv,
                "deepguard_batch_report.csv",
                "text/csv"
            )

# ==========================================
# TAB 4: HIỂU MÔ HÌNH (XAI)
# ==========================================
with tab4:
    st.header("🧠 Hiểu cách Mô hình hoạt động")
    st.markdown("""
    DeepGuard không phải là một hộp đen. Chúng tôi tích hợp **Explainable AI (XAI)** để bạn biết chính xác tại sao mô hình lại đưa ra quyết định đó.
    """)
    
    st.subheader("1. Grad-CAM là gì?")
    st.info("""
    **Gradient-weighted Class Activation Mapping (Grad-CAM)** là một kỹ thuật dùng để "nhìn xuyên" vào suy nghĩ của các mạng nơ-ron tích chập (CNN). 
    Nó sử dụng đạo hàm ngược từ nhãn dự đoán để làm nổi bật (highlight) những vùng trên ảnh mà AI đã dựa vào đó để ra quyết định.
    - **Màu đỏ/vàng**: Mức độ chú ý cao. Thường trùng khớp với những vết bóp méo, viền mờ do thuật toán Deepfake để lại quanh mắt, mũi, miệng.
    - **Màu xanh đậm**: Mức độ chú ý thấp.
    """)
    
    st.subheader("2. Cấu trúc Mô hình")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        **Kiến trúc:**
        - **Backbone**: `EfficientNet-B4` (Pretrained trên ImageNet)
        - **Classifier Head**: `GlobalAvgPool -> Dropout(0.4) -> Dense(512) -> ReLU -> Dropout(0.3) -> Dense(1)`
        - **Loss Function**: `Focal Loss` (giúp tập trung vào các ca khó dự đoán)
        """)
    with col_info2:
        st.markdown("""
        **Pipeline Xử lý:**
        1. **Face Detection**: Phát hiện và cắt khuôn mặt sử dụng `MTCNN`.
        2. **Preprocessing**: Thêm padding 20%, resize về `224x224`, normalize.
        3. **Inference**: Chạy qua mạng CNN, kích hoạt hàm `Sigmoid` cho ra xác suất cuối cùng.
        """)
        
    st.markdown("---")
    st.caption("Dự án được thực hiện nhằm mục đích nghiên cứu và nâng cao nhận thức về an toàn thông tin hình ảnh.")


# ==========================================
# TAB 5: SELF TEST
# ==========================================
with tab5:
    st.header("🧪 Tự kiểm thử với ảnh thật/giả công khai")
    st.markdown("Chạy bộ ảnh trong `data/external_test` qua API hiện tại để phát hiện lệch nhãn, threshold hoặc preprocessing.")

    sample_root = Path("data/external_test")
    sample_paths = []
    for label_name in ["real", "fake"]:
        label_dir = sample_root / label_name
        if label_dir.exists():
            sample_paths.extend(sorted([
                p for p in label_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]))

    st.caption(f"Đã tìm thấy {len(sample_paths)} ảnh self-test trong {sample_root}.")

    if sample_paths:
        preview_cols = st.columns(min(4, len(sample_paths)))
        for idx, sample_path in enumerate(sample_paths[:4]):
            with preview_cols[idx % len(preview_cols)]:
                st.image(str(sample_path), caption=f"{sample_path.parent.name}/{sample_path.name}", width='stretch')

    if st.button("Chạy self-test qua API", type="primary", disabled=not api_status or not sample_paths):
        rows = []
        progress = st.progress(0)
        for idx, sample_path in enumerate(sample_paths):
            expected = "REAL" if sample_path.parent.name.lower() == "real" else "FAKE"
            try:
                with open(sample_path, "rb") as f:
                    files = {"file": (sample_path.name, f, "image/jpeg")}
                    response = requests.post(
                        f"{API_BASE_URL}/predict/image",
                        files=files,
                        data={"threshold": threshold, "return_heatmap": "false"},
                        timeout=45,
                    )
                data = response.json()
                prob_fake = data.get("probability_fake", data.get("confidence", 0.0))
                predicted = data.get("label", "FAKE" if prob_fake >= threshold else "REAL")
                rows.append({
                    "file": str(sample_path),
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "probability_fake": round(prob_fake, 4),
                    "confidence": round(data.get("confidence", max(prob_fake, 1 - prob_fake)), 4),
                    "face_detected": data.get("face_detected"),
                })
            except Exception as e:
                rows.append({
                    "file": str(sample_path),
                    "expected": expected,
                    "predicted": "ERROR",
                    "correct": False,
                    "probability_fake": None,
                    "confidence": None,
                    "face_detected": None,
                    "error": str(e),
                })
            progress.progress((idx + 1) / len(sample_paths))

        df_self = pd.DataFrame(rows)
        st.dataframe(df_self, width='stretch')
        valid = df_self[df_self["predicted"].isin(["REAL", "FAKE"])]
        if not valid.empty:
            st.metric("Self-test accuracy", f"{valid['correct'].mean():.1%}")
        st.download_button(
            "Tải CSV self-test",
            df_self.to_csv(index=False).encode("utf-8"),
            "deepguard_external_self_test.csv",
            "text/csv",
        )
