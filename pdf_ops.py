import io
from dataclasses import dataclass
from typing import Callable, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SepiaOptions:
    intensity: float  # 0..1
    dpi: int
    image_format: str = "JPEG"  # JPEG or PNG
    jpeg_quality: int = 85
    background_only: bool = True
    background_threshold: float = 0.80


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def apply_sepia_np(rgb_img: Image.Image, intensity: float) -> Image.Image:
    """Fast sepia using NumPy (vectorized).

    intensity=0 keeps original; intensity=1 is full sepia.
    """
    intensity = _clamp01(intensity)

    if rgb_img.mode not in ("RGB", "RGBA"):
        rgb_img = rgb_img.convert("RGBA")

    if rgb_img.mode == "RGBA":
        rgb = rgb_img.convert("RGB")
        alpha = rgb_img.getchannel("A")
    else:
        rgb = rgb_img
        alpha = None

    arr = np.asarray(rgb, dtype=np.float32)

    # Sepia transform matrix
    sepia = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )

    sep = arr @ sepia.T
    sep = np.clip(sep, 0, 255)

    out = arr * (1.0 - intensity) + sep * intensity
    out = np.clip(out, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(out, mode="RGB")

    if alpha is not None:
        out_rgba = out_img.convert("RGBA")
        out_rgba.putalpha(alpha)
        return out_rgba

    return out_img


def apply_sepia_background_np(
    rgb_img: Image.Image,
    intensity: float,
    threshold: float = 0.80,
    bg_rgb: tuple[int, int, int] = (244, 236, 216),
) -> Image.Image:
    """Apply a sepia *background tint* while preserving dark pixels (text).

    This works well for typical black-text-on-white PDFs.

    threshold controls where tint starts (0..1). Higher threshold preserves more.
    """
    intensity = _clamp01(intensity)
    threshold = _clamp01(threshold)

    if rgb_img.mode not in ("RGB", "RGBA"):
        rgb_img = rgb_img.convert("RGBA")

    if rgb_img.mode == "RGBA":
        rgb = rgb_img.convert("RGB")
        alpha = rgb_img.getchannel("A")
    else:
        rgb = rgb_img
        alpha = None

    arr = np.asarray(rgb, dtype=np.float32)

    # Luminance in [0,1]
    y = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) / 255.0

    # Smooth mask: 0 below threshold, ramps to 1 at white
    denom = max(1e-6, 1.0 - threshold)
    m = np.clip((y - threshold) / denom, 0.0, 1.0) * intensity
    m = m[..., None]  # broadcast to RGB

    bg = np.array(bg_rgb, dtype=np.float32)[None, None, :]
    out = arr * (1.0 - m) + bg * m
    out = np.clip(out, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(out, mode="RGB")
    if alpha is not None:
        out_rgba = out_img.convert("RGBA")
        out_rgba.putalpha(alpha)
        return out_rgba
    return out_img


def pdf_to_sepia_pdf(
    pdf_bytes: bytes,
    options: SepiaOptions,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """Convert a PDF to a sepia-toned, image-based PDF.

    progress_cb(current_page_1_based, total_pages)
    """
    intensity = _clamp01(options.intensity)
    dpi = int(options.dpi)

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    total = src.page_count

    for idx in range(total):
        page = src.load_page(idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if options.background_only:
            pil = apply_sepia_background_np(pil, intensity, threshold=float(options.background_threshold))
        else:
            pil = apply_sepia_np(pil, intensity)

        img_io = io.BytesIO()
        fmt = (options.image_format or "JPEG").upper()
        save_kwargs = {}
        if fmt == "JPEG":
            save_kwargs["quality"] = int(options.jpeg_quality)
            save_kwargs["optimize"] = True
        pil.save(img_io, format=fmt, **save_kwargs)
        img_bytes = img_io.getvalue()

        rect = page.rect
        out_page = out.new_page(width=rect.width, height=rect.height)
        out_page.insert_image(rect, stream=img_bytes)

        if progress_cb is not None:
            progress_cb(idx + 1, total)

    result = out.tobytes(garbage=4, deflate=True)
    out.close()
    src.close()
    return result
