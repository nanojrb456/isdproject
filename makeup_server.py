"""
FastAPI server wrapping the 68-point makeup pipeline.

POST /makeup
Request JSON:
{
  "image_base64": "<base64-encoded image (JPEG/PNG)>",
  "style": "simple" | "glam",
  "lip_opacity": float,
  "shadow_opacity": float,
  "blush_opacity": float,
  "highlight_opacity": float,
  "lip_color": "#RRGGBB" | "r,g,b",
  "shadow_color": "#RRGGBB" | "r,g,b",
  "blush_color": "#RRGGBB" | "r,g,b",
  "highlight_color": "#RRGGBB" | "r,g,b",
  "rotate": 0 | 90 | 180 | 270,
  "mirror": true | false
}

Response JSON:
{
  "image_base64": "<base64-encoded PNG output>",
  "width": int,
  "height": int
}
"""

import base64
import io
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageOps


from virtual_makeup_68 import (
    MODEL_PATH,
    build_style,
    orient_frame,
    apply_makeup,
    parse_color,
    STYLES,
    CASCADE_PATH,
)



app = FastAPI(title="Virtual Makeup 68pt", version="1.0")

model_cache: Optional[tf.keras.Model] = None
cascade_cache: Optional[cv2.CascadeClassifier] = None


class MakeupRequest(BaseModel):
    image_base64: str
    style: str = Field("simple", description="Preset style")
    lip_opacity: Optional[float] = None
    shadow_opacity: Optional[float] = None
    blush_opacity: Optional[float] = None
    highlight_opacity: Optional[float] = None
    lip_color: Optional[str] = None
    shadow_color: Optional[str] = None
    blush_color: Optional[str] = None
    highlight_color: Optional[str] = None
    rotate: int = Field(0, description="0,90,180,270")
    mirror: bool = Field(False, description="mirror input (front cam)")
    debug: bool = Field(False, description="if true, draw bbox + landmarks on output")



from PIL import Image, ImageOps

def decode_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = ImageOps.exif_transpose(img)  # ✅ مهم
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode_image(bgr: np.ndarray) -> str:
    success, png = cv2.imencode(".png", bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return base64.b64encode(png.tobytes()).decode("utf-8")


def load_model_and_cascade():
    global model_cache, cascade_cache
    if model_cache is None:
        model_cache = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if cascade_cache is None:
        cascade_cache = cv2.CascadeClassifier(str(CASCADE_PATH))
    return model_cache, cascade_cache


def build_custom_style(req: MakeupRequest):
    class Args:
        def __init__(self, req: MakeupRequest):
            self.style = req.style if req.style in STYLES else "simple"
            self.lip_color = req.lip_color
            self.shadow_color = req.shadow_color
            self.blush_color = req.blush_color
            self.highlight_color = req.highlight_color
    args = Args(req)
    style = build_style(args)

    # Override opacities if provided
    if req.lip_opacity is not None:
        style["lipstick"]["opacity"] = float(req.lip_opacity)
    if req.shadow_opacity is not None:
        style["eyeshadow"]["opacity"] = float(req.shadow_opacity)
    if req.blush_opacity is not None:
        style["blush"]["opacity"] = float(req.blush_opacity)
    if req.highlight_opacity is not None:
        style["highlighter"]["opacity"] = float(req.highlight_opacity)
    return style


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/makeup")
def apply_makeup_endpoint(req: MakeupRequest):
    bgr = decode_image(req.image_base64)
    print(f"[debug] after exif_transpose decode: {bgr.shape}")
    model, cascade = load_model_and_cascade()
    style = build_custom_style(req)

    oriented = bgr
    if req.rotate != 0 or req.mirror:
        try:
            oriented = orient_frame(bgr, rotation=req.rotate, mirror=req.mirror)
        except ValueError:
            raise HTTPException(status_code=400, detail="rotate must be 0,90,270")
        print(f"[debug] after orient_frame rotate={req.rotate} mirror={req.mirror}: {oriented.shape}")

    # Keep everything in the oriented coordinate system (no deorient) so landmarks and bbox stay aligned.
    out = apply_makeup(oriented, model, cascade, style, debug=req.debug)
    print(f"[debug] output frame shape: {out.shape}")

    return {
        "image_base64": encode_image(out),
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("makeup_server:app", host="0.0.0.0", port=8000, reload=False)
