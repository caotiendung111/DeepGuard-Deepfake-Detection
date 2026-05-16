"""
Download public real/fake face samples and run a DeepGuard self-test.

Modes:
  api      Download samples, then call the running FastAPI endpoint.
  local    Download samples, then load the checkpoint directly.
  download Download samples only.

Example:
  python scripts/external_self_test.py --mode api --api-url http://localhost:8000
  python scripts/external_self_test.py --mode local --checkpoint models/checkpoints/best_model.pth
"""
import argparse
import csv
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_CONTEXT = None


SAMPLES = [
    {
        "id": "real_tomas",
        "label": 0,
        "class_name": "real",
        "commons_file": "1994fotoTomas.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:1994fotoTomas.jpg",
    },
    {
        "id": "real_luis_perez",
        "label": 0,
        "class_name": "real",
        "commons_file": "Foto de Luis Francsico Perez Pereyra.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Foto_de_Luis_Francsico_Perez_Pereyra.jpg",
    },
    {
        "id": "real_osni",
        "label": 0,
        "class_name": "real",
        "commons_file": "Osni do Amparo.png",
        "source_page": "https://commons.wikimedia.org/wiki/File:Osni_do_Amparo.png",
    },
    {
        "id": "real_camera",
        "label": 0,
        "class_name": "real",
        "commons_file": "A Man with a Camera.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:A_Man_with_a_Camera.jpg",
    },
    {
        "id": "fake_stylegan2_3",
        "label": 1,
        "class_name": "fake",
        "commons_file": "StyleGAN2 Example 3.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:StyleGAN2_Example_3.jpg",
    },
    {
        "id": "fake_tpdne",
        "label": 1,
        "class_name": "fake",
        "commons_file": "This Person Does Not Exist example.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:This_Person_Does_Not_Exist_example.jpg",
    },
    {
        "id": "fake_gan_white_girl",
        "label": 1,
        "class_name": "fake",
        "commons_file": "GAN deepfake white girl.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:GAN_deepfake_white_girl.jpg",
    },
    {
        "id": "fake_woman_1",
        "label": 1,
        "class_name": "fake",
        "commons_file": "Woman 1.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Woman_1.jpg",
    },
]


def commons_redirect_url(filename: str, width: int = 768) -> str:
    quoted = urllib.parse.quote(filename)
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quoted}?width={width}"


def extension_for(sample: dict) -> str:
    suffix = Path(sample["commons_file"]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def download_sample(
    sample: dict,
    output_root: Path,
    width: int,
    retries: int = 3,
) -> tuple[Path, str | None]:
    class_dir = output_root / sample["class_name"]
    class_dir.mkdir(parents=True, exist_ok=True)
    out_path = class_dir / f"{sample['id']}{extension_for(sample)}"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path, None

    url = commons_redirect_url(sample["commons_file"], width=width)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DeepGuard external self-test/1.0"},
    )
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                out_path.write_bytes(response.read())
            return out_path, None
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP Error {exc.code}: {exc.reason}"
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            break
    return out_path, last_error


def build_multipart(image_path: Path, threshold: float) -> tuple[bytes, str]:
    boundary = f"----DeepGuardBoundary{int(time.time() * 1000)}"
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    file_bytes = image_path.read_bytes()
    chunks = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="threshold"\r\n\r\n',
        f"{threshold}\r\n".encode(),
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="return_heatmap"\r\n\r\n',
        b"false\r\n",
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode(),
        f"Content-Type: {mime}\r\n\r\n".encode(),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(chunks), boundary


def predict_via_api(api_url: str, image_path: Path, threshold: float) -> tuple[dict | None, str | None]:
    body, boundary = build_multipart(image_path, threshold)
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/predict/image",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": "DeepGuard external self-test/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, str(exc)


def predict_local(image_path: Path, threshold: float, checkpoint: str, config: str) -> tuple[dict | None, str | None]:
    global _LOCAL_CONTEXT
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        from src.inference.face_detector import FaceDetector
        from src.inference.model_loader import load_detector_checkpoint
        from src.inference.predictor import predict_probability
        from src.utils.config import load_config
    except ImportError as exc:
        return None, f"Missing local inference dependency: {exc}"

    try:
        if _LOCAL_CONTEXT is None:
            cfg = load_config(config)
            cfg.checkpoint_path = checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _ = load_detector_checkpoint(checkpoint, cfg, device)
            detector = FaceDetector(device=str(device), face_size=cfg.image_size)
            _LOCAL_CONTEXT = (cfg, device, model, detector)
        cfg, device, model, detector = _LOCAL_CONTEXT
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        boxes = detector.detect_boxes(image_rgb)
        if boxes:
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            face_rgb = detector._crop(image_rgb, boxes[int(np.argmax(areas))])
            face_detected = True
        else:
            face_rgb = cv2.resize(image_rgb, (cfg.image_size, cfg.image_size))
            face_detected = False

        prob_fake, tta_probs = predict_probability(
            model=model,
            image_rgb=face_rgb,
            image_size=cfg.image_size,
            device=device,
            use_tta=cfg.inference_tta,
        )
        is_fake = prob_fake >= threshold
        return {
            "label": "FAKE" if is_fake else "REAL",
            "is_fake": is_fake,
            "probability_fake": prob_fake,
            "probability_real": 1.0 - prob_fake,
            "confidence": max(prob_fake, 1.0 - prob_fake),
            "threshold": threshold,
            "face_detected": face_detected,
            "tta_probabilities": tta_probs,
        }, None
    except Exception as exc:
        return None, str(exc)


def result_row(sample: dict, path: Path, prediction: dict | None, error: str | None) -> dict:
    expected = "FAKE" if sample["label"] == 1 else "REAL"
    predicted = prediction.get("label") if prediction else None
    return {
        "sample_id": sample["id"],
        "expected_label": expected,
        "predicted_label": predicted or "",
        "correct": "" if predicted is None else str(predicted == expected),
        "probability_fake": "" if not prediction else prediction.get("probability_fake", ""),
        "confidence": "" if not prediction else prediction.get("confidence", ""),
        "threshold": "" if not prediction else prediction.get("threshold", ""),
        "face_detected": "" if not prediction else prediction.get("face_detected", ""),
        "local_path": str(path),
        "source_page": sample["source_page"],
        "error": error or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download external samples and test DeepGuard.")
    parser.add_argument("--mode", choices=["api", "local", "download"], default="api")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pth")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output-dir", default="reports/external_self_test")
    parser.add_argument("--image-dir", default="data/external_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in SAMPLES:
        path, download_error = download_sample(sample, image_dir, args.width)
        time.sleep(args.sleep)
        if download_error:
            rows.append(result_row(sample, path, None, f"download: {download_error}"))
            continue

        prediction = None
        error = None
        if args.mode == "api":
            prediction, error = predict_via_api(args.api_url, path, args.threshold)
        elif args.mode == "local":
            prediction, error = predict_local(path, args.threshold, args.checkpoint, args.config)

        rows.append(result_row(sample, path, prediction, error))

    csv_path = output_dir / "external_self_test_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row["correct"] in {"True", "False"}]
    summary = {
        "mode": args.mode,
        "threshold": args.threshold,
        "samples": len(rows),
        "completed_predictions": len(completed),
        "accuracy": (
            sum(row["correct"] == "True" for row in completed) / len(completed)
            if completed else None
        ),
        "results_csv": str(csv_path),
    }
    (output_dir / "external_self_test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
