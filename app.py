# app.py — Pixelarc backend (Render / cloud version)
#
# Uses the ai-forever/Real-ESRGAN PyTorch package instead of the Windows
# ncnn-vulkan binary. This version runs on CPU — no GPU required — which
# is what Render's free tier provides.
#
# Same 4 endpoints as your local backend so the mobile app needs zero
# changes — just update src/config.js to point at the Render URL.

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io, os, base64, traceback, torch

app = Flask(__name__)
CORS(app)

# ── Model setup ──────────────────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

MODEL = None
AI_AVAILABLE = False

def load_model():
    global MODEL, AI_AVAILABLE
    try:
        from RealESRGAN import RealESRGAN
        print("  Loading RealESRGAN x4 weights...")
        model = RealESRGAN(DEVICE, scale=4)
        model.load_weights(
            os.path.join(WEIGHTS_DIR, "RealESRGAN_x4.pth"),
            download=True
        )
        MODEL = model
        AI_AVAILABLE = True
        print("  ✅ Model loaded successfully")
    except Exception as e:
        print(f"  ⚠️  Model load failed: {e}")
        AI_AVAILABLE = False

load_model()


def run_upscale(pil_img, scale):
    orig_w, orig_h = pil_img.size

    # Cap input size — free tier has only 512MB RAM. Memory usage scales
    # with pixels squared, so halving MAX_DIM quarters the memory needed.
    # 512px input → 2048px output at 4x, which is still a clear improvement
    # and fits comfortably within the RAM ceiling.
    MAX_DIM = 512
    work_w, work_h = orig_w, orig_h
    if max(orig_w, orig_h) > MAX_DIM:
        ratio = MAX_DIM / max(orig_w, orig_h)
        work_w, work_h = int(orig_w * ratio), int(orig_h * ratio)
        pil_img = pil_img.resize((work_w, work_h), Image.LANCZOS)
        print(f"  Resized {orig_w}x{orig_h} → {work_w}x{work_h}")

    result_img = None
    method = "pillow"

    if AI_AVAILABLE:
        try:
            with torch.no_grad():
                result_img = MODEL.predict(pil_img).convert("RGB")

            # Model always outputs 4x — resize down if 2x/3x was requested
            if scale != 4:
                result_img = result_img.resize(
                    (work_w * scale, work_h * scale), Image.LANCZOS
                )
            method = "realesrgan-pytorch"

        except Exception as e:
            print(f"  ⚠️  Inference failed: {e}")
            result_img = None

    if result_img is None:
        result_img = pil_img.resize((work_w * scale, work_h * scale), Image.LANCZOS)
        method = "pillow_fallback"

    return result_img, method, orig_w, orig_h


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ai_available": AI_AVAILABLE,
        "device": str(DEVICE),
        "engine": "realesrgan-pytorch",
    })


@app.route("/api/metadata", methods=["POST"])
def metadata():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    try:
        img = Image.open(request.files["image"].stream)
        w, h = img.size
        return jsonify({
            "width": w, "height": h,
            "format": img.format or "UNKNOWN",
            "megapixels": round((w * h) / 1_000_000, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    scale = int(request.form.get("scale", 2))
    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
        result_img, method, orig_w, orig_h = run_upscale(pil_img, scale)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return jsonify({
            "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "method": method,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "new_width": result_img.width,
            "new_height": result_img.height,
            "scale": scale,
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/upscale-binary", methods=["POST"])
def upscale_binary():
    """Returns raw PNG bytes instead of base64 — used by the mobile app
    to avoid JS heap OOM on large images."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    scale = int(request.form.get("scale", 2))
    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
        result_img, method, orig_w, orig_h = run_upscale(pil_img, scale)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)

        response = send_file(buf, mimetype="image/png")
        response.headers["X-Method"]      = method
        response.headers["X-Orig-Width"]  = str(orig_w)
        response.headers["X-Orig-Height"] = str(orig_h)
        response.headers["X-New-Width"]   = str(result_img.width)
        response.headers["X-New-Height"]  = str(result_img.height)
        response.headers["X-Scale"]       = str(scale)
        response.headers["Access-Control-Expose-Headers"] = (
            "X-Method, X-Orig-Width, X-Orig-Height, X-New-Width, X-New-Height, X-Scale"
        )
        return response
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 PIXELARC API (PyTorch/CPU) → port {port}")
    app.run(debug=False, port=port, host="0.0.0.0")