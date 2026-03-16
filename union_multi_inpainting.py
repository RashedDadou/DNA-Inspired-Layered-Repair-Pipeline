# union_multi_inpainting.py
# ملف منفصل لـ inpainting باستخدام ControlNet Union + multi-control + blending يدوي

import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline
from controlnet_aux import OpenposeDetector, MidasDetector, ZoeDetector

# ─── الإعدادات الأساسية ──────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32

# Scheduler و VAE (يمكن تغييرهما حسب النموذج)
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
)

# تحميل ControlNet Union
controlnet = ControlNetModel_Union.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True
)

# الـ Pipeline الرئيسي
pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch_dtype,
    scheduler=scheduler,
    safety_checker=None
)
pipe = pipe.to(device)

# ─── الـ Processors للـ multi-control (مثال: pose + depth) ────────────────
processor_pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

# ─── دالة الـ wrapper الرئيسية (اللي صممناها مع تعديل بسيط) ─────────────
def union_img2img_with_mask(
    prompt: str,
    negative_prompt: str = "",
    image: Image.Image = None,          # الصورة الأصلية
    mask_image: Image.Image = None,     # الماسك
    control_images: list = None,        # قائمة control images
    control_scales: list = None,        # قائمة scales لكل control
    strength: float = 0.75,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = None,
) -> Image.Image:
    """
    Inpainting باستخدام Union Pipeline + multi-control + blending يدوي
    """
    if seed is None:
        seed = random.randint(0, 2147483647)
    generator = torch.Generator(device.type).manual_seed(seed)

    # توليد صورة كاملة جديدة باستخدام الـ controls
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_images,  # قائمة control images
        union_control=True,
        union_control_type=torch.Tensor(control_scales),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # blending يدوي باستخدام الماسك
    if image is not None and mask_image is not None:
        image = image.convert("RGB").resize(result.size)
        mask_np = np.array(mask_image.convert("L")) / 255.0
        result_np = np.array(result)
        orig_np = np.array(image)

        # blending بسيط + Gaussian blur خفيف على الحواف لو عايز
        blended_np = (mask_np[..., None] * result_np + (1 - mask_np[..., None]) * orig_np).astype(np.uint8)

        # تحسين الحواف اختياري
        mask_blur = cv2.GaussianBlur(mask_np.astype(np.float32), (5, 5), 0)
        blended_np = (mask_blur[..., None] * result_np + (1 - mask_blur[..., None]) * orig_np).astype(np.uint8)

        return Image.fromarray(blended_np)

    return result


# ─── مثال تشغيل سريع (اختبار) ────────────────────────────────────────
if __name__ == "__main__":
    # مسارات الصور (غيّرها حسب جهازك)
    source_img_path = "input.jpg"
    save_path = "output_union_inpainting.png"

    source_img = cv2.imread(source_img_path)

    # توليد controls (مثال: pose + depth)
    control_pose = processor_pose(source_img, hand_and_face=False, output_type='pil')
    control_depth = processor_zoe(source_img, output_type='pil')

    # تحجيم لـ 1024x1024 أو bucket مناسب
    target_size = (1024, 1024)
    control_pose = control_pose.resize(target_size)
    control_depth = control_depth.resize(target_size)

    # الـ prompt
    prompt = "masterpiece, best quality, highly detailed, realistic skin, vibrant colors"
    negative_prompt = "blurry, low quality, artifacts, deformed, bad anatomy, extra limbs"

    # النداء
    result = union_img2img_with_mask(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=Image.open(source_img_path).convert("RGB"),
        mask_image=Image.open("mask.png").convert("L"),  # ماسكك هنا
        control_images=[control_pose, control_depth],
        control_scales=[1.0, 0.8],  # scale لكل control
        strength=0.75,
        num_inference_steps=30,
        guidance_scale=7.5,
    )

    result.save(save_path)
    print(f"تم الحفظ في: {save_path}")
