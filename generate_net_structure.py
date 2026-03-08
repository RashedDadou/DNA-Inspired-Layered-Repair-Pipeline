"""
generate_net_structure.py

الطبقة التانية: ControlNet Structure
- إنتاج الشبكة الهيكلية (Net) داخل حدود الماسك بدقة عالية
- يستخدم ControlNet Union كافتراضي لأعلى جودة (يدعم tile + canny + lineart + depth)
- يحمي الـ pose/layout/depth ويمنع التشوهات العشوائية
"""

from typing import Literal, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

from ..filters.helpers import _ensure_rgb  # افتراض وجوده


def generate_net_structure(
    img: Image.Image,
    mask: Image.Image,
    control_type: Literal["union", "tile", "canny", "lineart", "depth"] = "union",
    net_strength: float = 0.68,         # 0.0–1.0 (أقل = أكثر تحكم بالصورة الأصلية)
    steps: int = 18,                    # عدد خطوات الـ inference
    guidance_scale: float = 7.2,        # قوة التوجيه
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_repo: str = "lllyasviel/sd-controlnet-union-sdxl-1.0",  # أفضل Union حاليًا
) -> Image.Image:
    """
    إنتاج الشبكة الهيكلية (Net) باستخدام ControlNet داخل حدود الماسك فقط

    Parameters:
        img: الصورة الأصلية (PIL Image)
        mask: ماسك المنطقة المراد ترميمها (L mode)
        control_type: نوع التحكم (union أفضل)
        net_strength: قوة تأثير الـ Net (0.5–0.8 مثالي للترميم)
        steps: عدد الخطوات (15–25 كافي)
        guidance_scale: قوة الـ prompt (7–9 جيد)
        device: cpu أو cuda
        model_repo: ريبو ControlNet Union (أو غيره)

    Returns:
        PIL.Image: الشبكة الهيكلية (Net) كصورة RGB
    """
    img = _ensure_rgb(img)
    mask = mask.convert("L")

    # تحميل ControlNet (مرة واحدة في الـ init لو كان كلاس)
    controlnet = ControlNetModel.from_pretrained(
        model_repo, torch_dtype=torch.float16
    ).to(device)

    # تحميل Pipeline (مرة واحدة في الـ init لو كان كلاس)
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    pipeline.enable_attention_slicing()
    pipeline.enable_model_cpu_offload()

    # تحضير control image حسب النوع
    if control_type == "canny":
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        control_img = Image.fromarray(edges).convert("RGB")
    elif control_type == "lineart":
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        lineart = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        control_img = Image.fromarray(lineart).convert("RGB")
    elif control_type == "depth":
        # placeholder – تحتاج MiDaS أو Depth Anything
        control_img = img  # fallback مؤقت
    else:
        # union أو tile → نستخدم الصورة مباشرة
        control_img = img

    # إنتاج الشبكة داخل الماسك
    net_image = pipeline(
        prompt="structural grid, clean lines, detailed architecture, high contrast edges",
        negative_prompt="blurry, noisy, low detail, artifacts, deformed structure",
        image=img,
        mask_image=mask,
        control_image=control_img,
        strength=net_strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # تلميع خفيف للشبكة (اختياري – يحسن الوضوح)
    net_image = ImageEnhance.Contrast(net_image).enhance(1.15)
    net_image = ImageEnhance.Sharpness(net_image).enhance(1.10)

    return net_image


# ────────────────────────────────────────────────
# اختبار سريع (للتجربة المباشرة)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image

    # افتراض وجود صورة وماسك
    input_img = Image.open("input.jpg")
    input_mask = Image.open("mask.jpg").convert("L")  # ماسك أبيض/أسود

    net = generate_net_structure(
        img=input_img,
        mask=input_mask,
        control_type="union",
        net_strength=0.68,
        steps=18,
        guidance_scale=7.2
    )

    net.save("generated_net.jpg")
    print("تم إنتاج الشبكة → generated_net.jpg")
