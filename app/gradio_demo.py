"""
DeepGuard — Gradio Demo UI
Interactive web demo for deepfake detection with Grad-CAM heatmap visualization.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import requests
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DEMO_DESCRIPTION = """
# 🛡️ DeepGuard — Deepfake Detection System

**Upload an image or video** to detect whether it's **real** or a **deepfake**.

The system uses **EfficientNet-B4** trained on FaceForensics++ with Grad-CAM
visualization to highlight suspicious facial regions.

> ⚠️ **Note**: For best results, use clear face images/videos with good lighting.
"""

CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    font-family: 'Inter', sans-serif;
}
.result-real { 
    background: linear-gradient(90deg, #0d6b3e, #11a855);
    color: white; border-radius: 12px; padding: 16px; text-align: center;
    font-size: 1.5rem; font-weight: bold;
}
.result-fake {
    background: linear-gradient(90deg, #7b0d0d, #ef233c);
    color: white; border-radius: 12px; padding: 16px; text-align: center;
    font-size: 1.5rem; font-weight: bold;
}
.metric-box {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
    padding: 12px; margin: 8px 0;
}
"""


# ── Prediction Functions ───────────────────────────────────────────────────────
def predict_image_api(image: Image.Image, threshold: float, return_heatmap: bool):
    """Send image to FastAPI and return formatted results."""
    if image is None:
        return (
            gr.update(value="<p style='color:gray'>No image provided</p>", visible=True),
            None, "", "", ""
        )

    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/image",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            data={"threshold": threshold, "return_heatmap": str(return_heatmap).lower()},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return (
            gr.update(value="⚠️ API not running. Start with: `make api`", visible=True),
            None, "", "", ""
        )
    except Exception as e:
        return (
            gr.update(value=f"❌ Error: {str(e)}", visible=True),
            None, "", "", ""
        )

    label = data["label"]
    is_fake = data["is_fake"]
    prob_fake = data["probability_fake"]
    confidence = data["confidence"]
    elapsed = data["processing_time_ms"]

    label_css = "result-fake" if is_fake else "result-real"
    emoji = "🚨 DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC IMAGE"
    label_html = f'<div class="{label_css}">{emoji}</div>'

    summary = (
        f"**Verdict:** {label}\n\n"
        f"**P(FAKE):** {prob_fake:.1%}\n\n"
        f"**P(REAL):** {data['probability_real']:.1%}\n\n"
        f"**Confidence:** {confidence:.1%}\n\n"
        f"**Inference time:** {elapsed:.1f}ms"
    )

    # Decode heatmap if available
    heatmap_img = None
    if return_heatmap and data.get("heatmap_base64"):
        import base64
        heatmap_bytes = base64.b64decode(data["heatmap_base64"])
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

    prob_bar = f"P(FAKE): {prob_fake:.1%}"
    conf_bar = f"Confidence: {confidence:.1%}"

    return label_html, heatmap_img, summary, prob_bar, conf_bar


def predict_video_api(video_path: str, threshold: float, n_frames: int):
    """Send video to FastAPI and return results."""
    if video_path is None:
        return "No video provided", "", None

    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                f"{API_BASE_URL}/predict/video",
                files={"file": (Path(video_path).name, f, "video/mp4")},
                data={"threshold": threshold, "n_frames": n_frames},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return "⚠️ API not running. Start with: `make api`", "", None
    except Exception as e:
        return f"❌ Error: {str(e)}", "", None

    label = data["label"]
    is_fake = data["is_fake"]
    emoji = "🚨 DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC VIDEO"
    fake_ratio = data["fake_frame_ratio"]
    n_analyzed = data["n_frames_analyzed"]
    elapsed = data["processing_time_ms"]

    verdict = f"""## {emoji}

| Metric | Value |
|--------|-------|
| **Verdict** | {label} |
| **P(FAKE)** | {data['probability_fake']:.1%} |
| **Fake Frame Ratio** | {fake_ratio:.1%} ({int(fake_ratio * n_analyzed)}/{n_analyzed} frames) |
| **Frames Analyzed** | {n_analyzed} |
| **Inference Time** | {elapsed:.0f}ms |
"""

    # Per-frame probability bar chart
    frame_probs = data.get("frame_probabilities", [])
    plot = None
    if frame_probs:
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            colors = ["#ef233c" if p >= threshold else "#00b4d8" for p in frame_probs]
            ax.bar(range(len(frame_probs)), frame_probs, color=colors, alpha=0.85)
            ax.axhline(y=threshold, color="white", linestyle="--", linewidth=1.5, label=f"Threshold={threshold}")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frame Index", color="white")
            ax.set_ylabel("P(FAKE)", color="white")
            ax.tick_params(colors="white")
            ax.set_title("Per-Frame Deepfake Probability", color="white", fontweight="bold")
            ax.legend(facecolor="#16213e", labelcolor="white")
            plt.tight_layout()
            plot = fig
        except Exception:
            pass

    return verdict, "", plot


# ── Build Gradio Interface ─────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="DeepGuard — Deepfake Detection",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif"],
        ),
    ) as demo:
        gr.Markdown(DEMO_DESCRIPTION)

        with gr.Tabs():
            # ── IMAGE TAB ──────────────────────────────────────────────────────
            with gr.TabItem("🖼️ Image Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            type="pil",
                            label="Upload Face Image",
                            height=300,
                        )
                        with gr.Row():
                            threshold_slider = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="Decision Threshold",
                                info="Higher = stricter fake detection"
                            )
                        return_heatmap = gr.Checkbox(
                            label="Generate Grad-CAM Heatmap",
                            value=True,
                            info="Highlights regions influencing the decision"
                        )
                        img_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        img_label_html = gr.HTML(label="Verdict")
                        with gr.Row():
                            img_prob_text = gr.Textbox(label="Probability", interactive=False)
                            img_conf_text = gr.Textbox(label="Confidence", interactive=False)
                        img_summary = gr.Markdown(label="Analysis Summary")
                        img_heatmap = gr.Image(label="Grad-CAM Heatmap", height=250)

                img_btn.click(
                    fn=predict_image_api,
                    inputs=[img_input, threshold_slider, return_heatmap],
                    outputs=[img_label_html, img_heatmap, img_summary, img_prob_text, img_conf_text],
                )

                gr.Examples(
                    examples=[],
                    inputs=img_input,
                    label="Example Images (add your own)",
                )

            # ── VIDEO TAB ─────────────────────────────────────────────────────
            with gr.TabItem("🎥 Video Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="Upload Video (MP4/AVI)")
                        vid_threshold = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05,
                            label="Decision Threshold"
                        )
                        n_frames_slider = gr.Slider(
                            4, 64, value=16, step=4,
                            label="Frames to Analyze",
                            info="More frames = slower but more accurate"
                        )
                        vid_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        vid_verdict = gr.Markdown(label="Video Analysis Result")
                        vid_extra = gr.Markdown()
                        vid_plot = gr.Plot(label="Per-Frame Probability Timeline")

                vid_btn.click(
                    fn=predict_video_api,
                    inputs=[vid_input, vid_threshold, n_frames_slider],
                    outputs=[vid_verdict, vid_extra, vid_plot],
                )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "**DeepGuard v1.0** | Built with PyTorch · FastAPI · Gradio | "
            "[GitHub](https://github.com/yourusername/deepguard) | "
            "[API Docs](http://localhost:8000/docs)"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
