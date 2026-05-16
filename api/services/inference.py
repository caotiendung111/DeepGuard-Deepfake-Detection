"""
Core inference logic using ThreadPoolExecutor to prevent blocking the event loop.
"""
import base64
import time
from typing import Optional, Union

import cv2
import numpy as np

from ..schemas import ImagePredictionResponse
from ..dependencies import app_state
from src.inference.predictor import predict_probability


def _robust_fake_probability(face_prob: float, full_prob: float, face_detected: bool) -> tuple[float, str]:
    """
    Combine face-crop and full-image signals conservatively.

    The Kaggle image dataset is not guaranteed to match the API crop pipeline.
    For real-world selfies, a detector crop can amplify camera noise, skin
    smoothing, or compression artifacts. Treat strong face/full disagreement as
    uncertain instead of letting one crop produce an overconfident FAKE result.
    """
    if not face_detected:
        return full_prob, "Không phát hiện mặt rõ; dùng toàn ảnh nên kết quả kém chắc chắn."

    # A high face-crop score is meaningful for AI portraits. The full frame
    # often includes background/clothes and can dilute the artifact signal.
    # Keep this above the default threshold by a margin so borderline selfies
    # still fall into the conservative disagreement path.
    disagreement = abs(face_prob - full_prob)
    # High disagreement (e.g. face is 0.9 but full is 0.2) suggests 
    # either a very small deepfake patch or, more likely, an artifact 
    # like a filter or occlusion (finger, glasses) confusing the model.
    if disagreement >= 0.30:
        # Be conservative: if they disagree, lower the score
        return float(min(face_prob, full_prob)), (
            "Phát hiện sai lệch lớn giữa vùng mặt và toàn ảnh; "
            "có thể do vật cản (tay, kính) hoặc Filter làm đẹp gây nhiễu."
        )

    # Average them more evenly
    return float((0.5 * face_prob) + (0.5 * full_prob)), "Vùng mặt và toàn ảnh có sự đồng thuận cao."

def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy RGB image to base64 string."""
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str

def process_image_sync(
    image_bytes: bytes,
    threshold: Optional[float] = None,
    return_heatmap: bool = True,
    use_tta: Optional[Union[bool, str]] = None,
) -> ImagePredictionResponse:
    """Synchronous function to process a single image. Runs in threadpool."""
    t0 = time.time()
    
    # 1. Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format or corrupted file")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detector = app_state["face_detector"]
    model = app_state["model"]
    gradcam = app_state["gradcam"]
    cfg = app_state["config"]
    
    # 2. Detect face
    boxes = detector.detect_boxes(img_rgb)
    face_detected = bool(boxes)
    if boxes:
        # Take the largest face.
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        largest_box = boxes[np.argmax(areas)]
        face_resized = detector._crop(img_rgb, largest_box)
    else:
        # Keep the request usable for difficult images, but make the missing
        # face explicit in the response so callers can treat it carefully.
        face_resized = cv2.resize(img_rgb, (cfg.image_size, cfg.image_size))
    
    # 4. Inference. Score both the detected face crop and the full image. This
    # reduces false positives when the production crop differs from training.
    device = next(model.parameters()).device
    decision_threshold = cfg.threshold if threshold is None else threshold
    effective_tta = cfg.inference_tta if use_tta is None else use_tta
    use_amp = bool(getattr(cfg, "inference_amp", False))
    face_prob, tta_probs = predict_probability(
        model=model,
        image_rgb=face_resized,
        image_size=cfg.image_size,
        device=device,
        use_tta=effective_tta,
        use_amp=use_amp,
    )
    full_resized = cv2.resize(img_rgb, (cfg.image_size, cfg.image_size))
    full_prob, _ = predict_probability(
        model=model,
        image_rgb=full_resized,
        image_size=cfg.image_size,
        device=device,
        use_tta=False,
        use_amp=use_amp,
    )
    prob, analysis_note = _robust_fake_probability(face_prob, full_prob, face_detected)

    is_fake = prob >= decision_threshold
    label = "FAKE" if is_fake else "REAL"
    confidence = max(prob, 1.0 - prob)
    
    heatmap_b64 = None
    if return_heatmap and is_fake and gradcam is not None:
        # Generate gradcam
        try:
            combined_img = gradcam.generate(face_resized, target=0)
            heatmap_b64 = image_to_base64(combined_img)
        except Exception as e:
            # Fallback if gradcam fails
            print(f"GradCAM generation failed: {e}")
            
    t1 = time.time()
    processing_time_ms = (t1 - t0) * 1000
    
    return ImagePredictionResponse(
        label=label,
        is_fake=is_fake,
        probability_fake=prob,
        probability_real=1.0 - prob,
        confidence=confidence,
        threshold=decision_threshold,
        face_detected=face_detected,
        processing_time_ms=processing_time_ms,
        heatmap_base64=heatmap_b64,
        tta_probabilities=tta_probs,
        face_probability_fake=face_prob,
        full_probability_fake=full_prob,
        analysis_note=analysis_note,
    )
