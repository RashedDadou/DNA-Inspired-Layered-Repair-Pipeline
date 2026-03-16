# test_pipeline_minimal.py
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device} | dtype: {dtype}")

print("جاري تحميل ControlNet Union...")
controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0",
    torch_dtype=dtype,
)

print("جاري تحميل الـ inpainting pipeline...")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",   # أو نموذج SD 1.5 inpaint أخف
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
)

pipe = pipe.to(device)

if device == "cuda":
    try:
        pipe.enable_model_cpu_offload()
        print("تم تفعيل cpu offload")
    except:
        print("cpu offload ما نفعش، هنشتغل بدون")

# اختبار بسيط جدًا (بدون صورة حقيقية حتى)
print("الـ pipeline تحمّل بنجاح! ✓")

# لو عايز تجرب generation بسيط (يحتاج صورة + ماسك)
# مثال:
# init_image = load_image("input.jpg").resize((512, 512))
# mask_image = Image.new("L", (512, 512), 0)  # كلها سودا = مفيش inpaint
# out = pipe(prompt="test", image=init_image, mask_image=mask_image, num_inference_steps=20).images[0]
