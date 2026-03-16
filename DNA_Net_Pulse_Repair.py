# DNA_Net_Pulse_Repair.py

"""
DNA_Repair_Pipeline.py

النهج الجديد: DNA-inspired Layer + ControlNet-guided Net
مع إسقاط نبضي DNA-inspired coloring (موجب/سالب) + طبقات لونية مخصصة
"""

import torch
print(torch.__version__)
print(torch.cuda.is_available())          # لازم True
print(torch.version.cuda)                 # لازم يطبع الـ CUDA version
print(torch.cuda.get_device_name(0))      # اسم الكارت

from typing import Optional, Literal, Tuple, List
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
import random
import colorsys

# Diffusers + ControlNet
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel as _ControlNetUnionModel
ControlNetUnionModel = _ControlNetUnionModel   # alias عشان تبقى تكتبها زي الأولfrom diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import StableDiffusionXLControlNetUnionInpaintPipeline

from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",   # أو stabilityai/stable-diffusion-xl-base-1.0
    controlnet=self.controlnet,
    torch_dtype=self.dtype,
    variant="fp16",
    safety_checker=None,
).to(self.device)



# ────────────────────────────────────────────────
# الجهاز والـ dtype
# ────────────────────────────────────────────────
from torch import device as torch_device, float16, float32

# ثم استخدمهم كده
device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
dtype = float16 if torch.cuda.is_available() else float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# ────────────────────────────────────────────────
# تحميل ControlNet Union
# ────────────────────────────────────────────────
print("جاري تحميل ControlNet Union...")

LOCAL_CONTROLNET_PATH = r"C:\Users\Rashed_Dadou\Desktop\SuperVisorSmartReporter\Rehabilitation Pipeline\update\Update"

try:
    controlnet = ControlNetUnionModel.from_pretrained(
        LOCAL_CONTROLNET_PATH,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    print("   → تم التحميل من المسار المحلي بنجاح")
except Exception as local_err:
    print(f"   → فشل التحميل المحلي: {local_err}")
    print("   → هنجرب التحميل من Hugging Face...")
    try:
        controlnet = ControlNetUnionModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0",
            torch_dtype=dtype,
            variant="fp16",
        )
        print("   → تم التحميل من Hugging Face بنجاح")
    except Exception as hf_err:
        raise RuntimeError(f"فشل تحميل ControlNet Union كليًا:\nLocal: {local_err}\nHF: {hf_err}")

controlnet = controlnet.to(device)
print(f"ControlNet موجود على: {next(controlnet.parameters()).device}")

# ────────────────────────────────────────────────
# تحميل Pipeline
# ────────────────────────────────────────────────
print("\nجاري تحميل Pipeline...")
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=dtype,
    variant="fp16",
    safety_checker=None,
)
pipe = pipe.to(device)

print("Pipeline loaded successfully! ✓")
print(f"Pipeline device: {pipe.device}")

# ────────────────────────────────────────────────
# 1. قائمة Menu – معيار الألوان الرقمي (DND style)
# ────────────────────────────────────────────────

DND_COLOR_MENU: dict[str, dict] = {
    "Fire": {
        "base_rgb": (220, 60, 30),
        "hex": "#DC3C1E",
        "element": "Fire",
        "energy": "high",
        "temperature": "hot",
        "mood": "aggression, passion, destruction"
    },
    "Ice": {
        "base_rgb": (80, 180, 255),
        "hex": "#50B4FF",
        "element": "Ice",
        "energy": "low",
        "temperature": "cold",
        "mood": "calm, control, preservation"
    },
    "Poison": {
        "base_rgb": (120, 40, 180),
        "hex": "#7828B4",
        "element": "Poison",
        "energy": "medium",
        "temperature": "neutral",
        "mood": "corruption, stealth, decay"
    },
    "Nature": {
        "base_rgb": (40, 180, 60),
        "hex": "#28B43C",
        "element": "Nature",
        "energy": "medium",
        "temperature": "warm",
        "mood": "growth, life, harmony"
    },
    "Shadow": {
        "base_rgb": (50, 30, 70),
        "hex": "#321E46",
        "element": "Shadow",
        "energy": "low",
        "temperature": "cold",
        "mood": "mystery, fear, void"
    },
    "Arcane": {
        "base_rgb": (160, 80, 220),
        "hex": "#A050DC",
        "element": "Arcane",
        "energy": "high",
        "temperature": "neutral",
        "mood": "magic, intellect, power"
    },
    "Radiant": {
        "base_rgb": (255, 240, 180),
        "hex": "#FFF0B4",
        "element": "Radiant",
        "energy": "high",
        "temperature": "hot",
        "mood": "holy, purity, light"
    }
}

class DndSeedColorEngine:
    """
    محرك لتوليد ومزج ألوان مستوحاة من نظام D&D مع لمسة DNA-inspired
    (طفرة خفيفة، تأثير طاقة عنصرية، مزج جيني)

    هذا الكلاس لا يحمل نماذج ثقيلة (ControlNet أو SD) → يركز على الألوان فقط
    """

    def __init__(
        self,
        color_menu: Optional[dict[str, dict]] = None,
        default_variation: float = 0.12,
        default_brightness_boost: float = 0.0,
        default_chaos_factor: float = 0.08,
        default_element_influence: float = 0.3,
    ):
        """
        تهيئة محرك الألوان D&D + DNA

        Args:
            color_menu: قاموس الألوان المخصص (اختياري، لو مش موجود يستخدم DND_COLOR_MENU الافتراضي)
            default_variation: نسبة التغيير العشوائي الافتراضية (±%)
            default_brightness_boost: زيادة/تقليل السطوع الافتراضي
            default_chaos_factor: قوة الطفرة العشوائية الافتراضية
            default_element_influence: قوة تأثير "الطاقة" بين العناصر
        """
        # استخدام القائمة المخصصة أو الافتراضية
        self.color_menu = color_menu if color_menu is not None else DND_COLOR_MENU

        # حفظ الإعدادات الافتراضية
        self.default_variation = default_variation
        self.default_brightness_boost = default_brightness_boost
        self.default_chaos_factor = default_chaos_factor
        self.default_element_influence = default_element_influence

        # قائمة العناصر المتاحة للاختيار العشوائي
        self.elements = list(self.color_menu.keys())

        print(f"DndSeedColorEngine جاهز | عدد العناصر: {len(self.elements)}")

    def generate_dnd_seed_color(
        element: str = "random",
        variation: float = 0.12,   # ±12% تغيير في القيم
        brightness_boost: float = 0.0
    ) -> Tuple[int, int, int]:
        """
        توليد لون Seed من قائمة DND أو عشوائي
        """
        if element == "random" or element not in DND_COLOR_MENU:
            element = random.choice(list(DND_COLOR_MENU.keys()))

        base = DND_COLOR_MENU[element]["base_rgb"]

        # إضافة variation جينية خفيفة
        r = int(base[0] * (1 + random.uniform(-variation, variation)))
        g = int(base[1] * (1 + random.uniform(-variation, variation)))
        b = int(base[2] * (1 + random.uniform(-variation, variation)))

        # تعديل السطوع
        r = int(r * (1 + brightness_boost))
        g = int(g * (1 + brightness_boost))
        b = int(b * (1 + brightness_boost))

        return tuple(np.clip([r, g, b], 0, 255).astype(int))

    def mix_dnd_seed_colors(
        color1: Tuple[int, int, int],
        color2: Optional[Tuple[int, int, int]] = None,
        ratio: float = 0.5,             # نسبة المزج (0.0 = color1 فقط، 1.0 = color2 فقط)
        element_influence: float = 0.3, # تأثير "الطاقة" بين العناصر
        chaos_factor: float = 0.08      # طفرة عشوائية خفيفة (DNA-style)
    ) -> Tuple[int, int, int]:
        """
        مزج ألوان بطريقة DNA-inspired (دمج + تأثير عنصري + طفرة خفيفة)
        """
        if color2 is None:
            color2 = generate_dnd_seed_color("random")

        # مزج خطي أساسي
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)

        # تأثير عنصري (DNA-inspired energy flow)
        if element_influence > 0:
            # افتراض أن color1 و color2 لهم "طاقة" مختلفة
            energy_diff = (sum(color1) - sum(color2)) / 765.0  # -1 → 1
            r += int(energy_diff * 40 * element_influence)
            g += int(-energy_diff * 30 * element_influence)
            b += int(energy_diff * 20 * element_influence)

        # طفرة جينية خفيفة (chaos)
        r += int(random.gauss(0, chaos_factor * 30))
        g += int(random.gauss(0, chaos_factor * 30))
        b += int(random.gauss(0, chaos_factor * 30))

        return tuple(np.clip([r, g, b], 0, 255).astype(int))

    def monitor_dnd_color_mix(
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        result: Tuple[int, int, int],
        ratio: float
    ) -> dict:
        """
        تقرير تحليلي عن عملية المزج
        """
        report = {}

        # حساب التوازن
        avg1 = sum(color1) / 3
        avg2 = sum(color2) / 3
        avg_res = sum(result) / 3

        report["brightness_balance"] = round(avg_res / ((avg1 * (1-ratio)) + (avg2 * ratio)), 3)
        report["dominant_element"] = "Fire" if result[0] > result[1] + result[2] else \
                                    "Nature" if result[1] > result[0] + result[2] else \
                                    "Arcane/Ice" if result[2] > result[0] + result[1] else "Neutral"

        report["energy_flow"] = "موجب قوي" if avg_res > (avg1 + avg2)/2 else "سالب متوازن" if avg_res < (avg1 + avg2)/2 else "متوازن"

        report["mutation_effect"] = f"طفرة لونية بنسبة ±{int(abs(avg_res - ((avg1 * (1-ratio)) + (avg2 * ratio))) * 100 / 255)}%"

        return report

    def add_colored_layers_on_net(
        net_image: Image.Image,
        mask: Image.Image,
        colors: List[tuple[int, int, int]] = [(255, 60, 60), (60, 255, 60), (60, 60, 255)],  # R, G, B
        blend_mode: Literal["density", "wave", "genetic_random"] = "density",
        base_opacity: float = 0.55,
    ) -> Image.Image:
        """
        إضافة طبقات لونية DNA-inspired على أضلاع الشبكة (Net) في منطقة الماسك فقط.
        لا تعدل الصورة الأصلية، بل ترجع طبقة شفافة جديدة يمكن دمجها بعدين.

        Parameters:
            net_image: الصورة التي تحتوي على الشبكة (Net) – يفضل تكون RGBA
            mask: ماسك L (0–255) يحدد المناطق المستهدفة
            colors: قائمة ألوان RGB للدمج (افتراضي: أحمر، أخضر، أزرق)
            blend_mode:
                "density"         → توزيع حسب كثافة الماسك (أحمر في المناطق القوية، أزرق في الضعيفة)
                "wave"            → دمج موجي حلزوني (DNA helix style)
                "genetic_random"  → مزيج عشوائي جيني خفيف لكل بكسل
            base_opacity: الشفافية الأساسية للطبقة (0.0 → 1.0)

        Returns:
            طبقة RGBA جديدة يمكن دمجها فوق الصورة الأصلية بـ Image.alpha_composite أو blend
        """
        # تحويل إلى مصفوفات numpy
        net_arr = np.array(net_image.convert("RGBA"))
        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

        # استخراج حواف الشبكة (Net edges) للرسم عليها لاحقًا
        gray = cv2.cvtColor(net_arr[..., :3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # إنشاء طبقة جديدة شفافة (RGBA)
        h, w = mask_arr.shape
        layer = np.zeros((h, w, 4), dtype=np.uint8)

        if blend_mode == "density":
            # كثافة مرتفعة → أحمر قوي، كثافة منخفضة → أزرق أكثر
            intensity = np.clip(mask_arr ** 1.5, 0.0, 1.0)  # تركيز على المناطق القوية
            weights = np.stack([
                intensity * 0.6,                     # أحمر أقوى
                intensity * 0.45,                    # أخضر متوسط
                intensity * 0.35                     # أزرق أقل
            ], axis=-1)
            weights = np.clip(weights, 0.0, 1.0)

            for i, color in enumerate(colors):
                layer[..., :3] += (np.array(color, dtype=np.uint16) * weights[..., i, None]).astype(np.uint8)

            layer[..., 3] = (mask_arr * 255 * base_opacity).astype(np.uint8)

        elif blend_mode == "wave":
            # موجة حلزونية DNA-like
            y, x = np.indices((h, w))
            wave = np.sin(0.05 * y + 0.2 * x + np.linspace(0, 40 * np.pi, w))
            wave = (wave + 1) / 2  # 0 → 1
            wave = np.clip(wave, 0.0, 1.0)

            # دمج الألوان مع الموجة
            mixed = np.zeros((h, w, 3), dtype=np.float32)
            for i, color in enumerate(colors):
                mixed += np.array(color, dtype=np.float32) * wave * (1.0 / len(colors))

            layer[..., :3] = np.clip(mixed, 0, 255).astype(np.uint8)
            layer[..., 3] = (mask_arr * 255 * base_opacity).astype(np.uint8)

        elif blend_mode == "genetic_random":
            # مزيج عشوائي جيني (Dirichlet distribution مثل نسب DNA)
            ratios = np.random.dirichlet([1.0, 1.0, 1.0], size=(h, w))  # shape: (h, w, 3)

            mixed = np.zeros((h, w, 3), dtype=np.float32)
            for i, color in enumerate(colors):
                mixed += np.array(color, dtype=np.float32) * ratios[..., i, None]

            layer[..., :3] = np.clip(mixed, 0, 255).astype(np.uint8)
            layer[..., 3] = (mask_arr * 255 * base_opacity).astype(np.uint8)

        # رسم حواف الشبكة بلون مميز (اختياري – يساعد في الرؤية)
        layer[edges > 0, :3] = (220, 220, 255)  # أزرق فاتح
        layer[edges > 0, 3] = 220               # شفافية عالية للخطوط

        return Image.fromarray(layer, mode="RGBA")

    def generate_net(
            self,
            img: Image.Image,
            mask: Image.Image,
            control_type: Literal["union", "tile", "canny", "lineart"] = "union",
            net_strength: float = 0.68,
            steps: int = 18,
        ) -> Image.Image:
            """إنتاج شبكة هيكلية عالية الجودة داخل الماسك"""
            img = _ensure_rgb(img)
            mask = mask.convert("L")

            control_img = img
            if control_type == "canny":
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 60, 180)
                control_img = Image.fromarray(edges).convert("RGB")
            elif control_type == "lineart":
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                lineart = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
                control_img = Image.fromarray(lineart).convert("RGB")

            net = self.pipeline(
                prompt="structural grid, clean edges, detailed architecture",
                negative_prompt="blurry, noisy, artifacts, low detail",
                image=img,
                mask_image=mask,
                control_image=control_img,
                strength=net_strength,
                num_inference_steps=steps,
                guidance_scale=7.2,
            ).images[0]

            return net

class DNANetPulseRepair:
    """
    محرك الإصلاح الرئيسي: يجمع بين الترميم الهندسي (ControlNet)
    + طبقات DNA + نبض لوني DNA-inspired + إصلاح لوني/هيكلي

    يعتمد على كائن DndSeedColorEngine لتوليد الألوان (يُمرر له خارجيًا)
    """

    def __init__(
        self,
        controlnet_model: str = "xinsir/controlnet-union-sdxl-1.0",
        sd_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        color_engine=None,
        enable_attention_slicing: bool = True,
        enable_cpu_offload: bool = True,
        variant: str = "fp16",
    ):
        """
        DNANetPulseRepair – محرك إصلاح DNA-inspired مع ControlNet Union
        تم تصميمه بشكل modular ليسهل التطوير المستقبلي (LoRA, IP-Adapter, custom pulses, إلخ)
        """

        # ────────────────────────────────────────────────
        # Section 1: التوثيق والـ Parameters (سهل التعديل)
        # ────────────────────────────────────────────────
        self.color_engine = color_engine

        # ────────────────────────────────────────────────
        # Section 2: Device & Dtype Setup
        # ────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = dtype

        print(f"Using device: {self.device} | dtype: {self.dtype}")

        # ────────────────────────────────────────────────
        # Section 3: Color Engine (اختياري)
        # ────────────────────────────────────────────────
        if self.color_engine is None:
            print("No color engine passed → سيتم استخدام DndSeedColorEngine الافتراضي لاحقًا")
        else:
            print("Color Engine تم ربطه بنجاح")

        # ────────────────────────────────────────────────
        # Section 4: ControlNet Union Loader (الأهم – محلي أولاً)
        # ────────────────────────────────────────────────
        LOCAL_CONTROLNET_PATH = r"C:\Users\Rashed_Dadou\Desktop\SuperVisorSmartReporter\Rehabilitation Pipeline\update\Update"

        print("\n=== تحميل ControlNet Union ===")
        try:
            self.controlnet = ControlNetUnionModel.from_pretrained(
                LOCAL_CONTROLNET_PATH,
                torch_dtype=self.dtype,
                variant=variant,
                use_safetensors=True,
                local_files_only=True,
            )
            print("✓ تم التحميل من المسار المحلي بنجاح")
        except Exception as local_err:
            print(f"⚠ فشل التحميل المحلي: {local_err}")
            print("→ جاري محاولة التحميل من Hugging Face...")
            self.controlnet = ControlNetUnionModel.from_pretrained(
                controlnet_model,
                torch_dtype=self.dtype,
                variant=variant,
            )
            print("✓ تم التحميل من Hugging Face")

        self.controlnet = self.controlnet.to(self.device)
        print(f"ControlNet موجود على: {next(self.controlnet.parameters()).device}")

        # ────────────────────────────────────────────────
        # Section 5: Pipeline Loader
        # ────────────────────────────────────────────────
        print("\n=== تحميل Pipeline ===")
        try:
            self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                sd_model,
                controlnet=self.controlnet,
                torch_dtype=self.dtype,
                variant=variant,
                safety_checker=None,
            )
            print("✓ Pipeline تم تحميله بنجاح")
        except Exception as e:
            print(f"⚠ فشل الـ base model: {e}")
            print("→ جاري استخدام inpainting fallback...")
            self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                controlnet=self.controlnet,
                torch_dtype=self.dtype,
                variant=variant,
                safety_checker=None,
            )
            print("✓ Pipeline (fallback) تم تحميله")

        self.pipeline = self.pipeline.to(self.device)

        # ────────────────────────────────────────────────
        # Section 6: Performance Optimizations
        # ────────────────────────────────────────────────
        if enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
            print("✓ تم تفعيل attention slicing")

        if enable_cpu_offload and self.device.type == "cuda":
            try:
                self.pipeline.enable_model_cpu_offload()
                print("✓ تم تفعيل model CPU offload")
            except Exception as e:
                print(f"⚠ CPU offload فشل: {e} (هنكمل بدون)")

        # ────────────────────────────────────────────────
        # Section 7: Internal State & Validation
        # ────────────────────────────────────────────────
        self._is_ready = True
        self._local_controlnet_path = LOCAL_CONTROLNET_PATH

        # ────────────────────────────────────────────────
        # Section 8: Final Ready Message
        # ────────────────────────────────────────────────
        print("\n✅ DNANetPulseRepair جاهز تماماً للعمل")
        print(f"   Device : {self.device}")
        print(f"   ControlNet : {self.controlnet.__class__.__name__}")
        print(f"   Pipeline   : {self.pipeline.__class__.__name__}")
        print("   يمكنك الآن استخدام .repair() أو .generate_net_structure() 🔥")


    def detect_dead_zones(
        self,
        img: Image.Image,
        method: str = "canny_dilate",       # "canny_dilate", "laplacian_var", "entropy"
        canny_low: int = 60,
        canny_high: int = 180,
        dilation_kernel_size: int = 9,      # حجم kernel التوسيع (أكبر = مناطق أوسع)
        threshold: float = 0.22,            # عتبة نهائية لتحويل إلى ماسك ثنائي
        min_area_ratio: float = 0.008,      # تجاهل المناطق الصغيرة جدًا (نسبة من مساحة الصورة)
    ) -> Image.Image:
        """
        كشف المناطق "الميتة" أو المنهارة / منخفضة التفاصيل بطريقة كلاسيكية سريعة ومستقرة

        الطريقة الافتراضية: Canny edges → dilation → threshold على كثافة الحواف

        Returns:
            ماسك PIL Image مود L (0 = سليم، 255 = منطقة تحتاج إصلاح)
        """
        # 1. تحويل إلى numpy + grayscale
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if method == "canny_dilate":
            # ─── أكثر الطرق موثوقية في معظم الحالات ───
            edges = cv2.Canny(gray, canny_low, canny_high)

            # توسيع الحواف لتغطية المناطق الفقيرة بالتفاصيل
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # عكس: المناطق قليلة الحواف = مناطق ميتة
            edge_density = dilated.astype(np.float32) / 255.0

            # كلما قلت كثافة الحواف → المنطقة أكثر "موتًا"
            dead_score = 1.0 - edge_density

            # تطبيع + عتبة
            dead_score = (dead_score - dead_score.min()) / (dead_score.max() - dead_score.min() + 1e-8)
            mask = (dead_score > threshold).astype(np.uint8) * 255

        elif method == "laplacian_var":
            # ─── طريقة بديلة: تباين لابلاسيان (جيد للـ blur / low-detail) ───
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = cv2.GaussianBlur(np.abs(lap), (5,5), 0)
            lap_var = lap_var / (lap_var.max() + 1e-8)

            # المناطق منخفضة التباين = ميتة
            dead_score = 1.0 - lap_var
            mask = (dead_score > threshold * 1.3).astype(np.uint8) * 255   # عتبة أعلى شوية

        elif method == "entropy":
            # ─── طريقة متقدمة نسبيًا: entropy محلي (بطيئة شوية) ───
            # تحتاج تنفيذ entropy محلي بـ skimage أو يدوي
            # حاليًا نتركها كـ placeholder
            mask = np.zeros_like(gray)

        else:
            raise ValueError(f"طريقة غير مدعومة: {method}")

        # 2. تنظيف الماسك
        # إزالة المناطق الصغيرة جدًا
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        # توسيع خفيف للاندماج الأفضل مع الإصلاح
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

        # 3. تحويل إلى PIL L mode
        mask_pil = Image.fromarray(mask).convert("L")

        return mask_pil


    def _ensure_rgb(img: Image.Image) -> Image.Image:
        """تحويل الصورة إلى RGB إذا لم تكن كذلك"""
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def repair_with_pulse_layer(
        self,
        img: Image.Image,
        prompt: str = "high quality, realistic details, vibrant colors",
        control_type: Literal["tile", "canny", "depth"] = "tile",
        pulse_steps: int = 5,
        blend_opacity: float = 0.65,
    ) -> Image.Image:
        """
        الدالة الرئيسية للترميم الطبقي النبضي
        """
        img = _ensure_rgb(img)

        # 1. اكتشاف المناطق المنهارة بدقة عالية
        mask = self.detect_dead_zones(img, control_type=control_type)

        # 2. إنشاء الطبقة الجديدة مع نبضات موجب/سالب
        repair_layer = self.create_repair_layer(
            img.size,
            pulse_steps=pulse_steps,
        )

        # 3. دمج الطبقة فقط على المناطق المنهارة
        masked_layer = Image.composite(repair_layer, Image.new("RGBA", img.size, (0,0,0,0)), mask)

        # 4. دمج مع الصورة الأصلية (مع opacity قابل للتحكم)
        result = Image.alpha_composite(img.convert("RGBA"), masked_layer)
        result = result.convert("RGB")

        # 5. تلميع خفيف نهائي
        result = ImageEnhance.Sharpness(result).enhance(1.12)
        result = ImageEnhance.Contrast(result).enhance(1.06)

        return result

    # ====================== 1. إنتاج الشبكة عبر ControlNet ======================
    def generate_net_structure(
        self,
        img: Image.Image,
        mask: Image.Image,
        control_type: str = "canny",           # الافتراضي الآن canny
        net_strength: float = 0.65,
        steps: int = 20,
        canny_low: int = 70,
        canny_high: int = 170,
        lineart_thickness: int = 2,            # للتحكم في سمك الخطوط إذا استخدمنا lineart
    ) -> Image.Image:
        """
        إنشاء طبقة شبكة هيكلية (net) باستخدام ControlNet
        الآن نعتمد بشكل أساسي على canny أو lineart لتجنب مشاكل Union
        """
        img = self._ensure_rgb(img)
        mask = mask.convert("L")

        # ─── تحضير صورة الـ control حسب النوع ───
        if control_type == "canny":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, canny_low, canny_high)
            # توسيع خفيف عشان الخطوط ما تكونش رفيعة أوي
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            control_img = Image.fromarray(edges).convert("RGB")

        elif control_type == "lineart":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            # طريقة بسيطة وسريعة نسبياً لخطوط ناعمة
            lineart = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=11 + lineart_thickness*2,
                C=2
            )
            # عكس الألوان عشان الخطوط تبقى سوداء على بيضاء (أفضل لـ ControlNet)
            lineart = 255 - lineart
            control_img = Image.fromarray(lineart).convert("RGB")

        elif control_type == "union":
            # الخيار القديم – نستخدمه فقط لو المستخدم طلب صراحة
            control_img = img

        else:
            raise ValueError(f"control_type غير مدعوم: {control_type}. الخيارات: canny, lineart, union")

        # ─── توليد الشبكة ───
        with torch.no_grad():
            result = self.pipeline(
                prompt="structural grid, clean architectural lines, technical blueprint style, high contrast edges",
                negative_prompt="blurry, noisy, low detail, artifacts, text, watermark",
                image=img,
                mask_image=mask,
                control_image=control_img,
                controlnet_conditioning_scale=net_strength,   # ← هنا بنستخدم الـ strength كـ conditioning scale
                num_inference_steps=steps,
                guidance_scale=7.0,
            ).images[0]

        return result

    # ====================== 2. DNA Layer شفافة أولى ======================
    def create_dna_base_layer(
        size: Tuple[int, int],
        base_color: Tuple[int, int, int] = (40, 120, 60),   # أخضر DNA خفيف
        opacity: float = 0.38,
    ) -> Image.Image:
        """
        إنشاء طبقة أساس DNA شفافة ثابتة (تُستخدم كخلفية/أساس للإحياء)

        Args:
            size: أبعاد الصورة (width, height)
            base_color: لون أساسي RGB
            opacity: درجة الشفافية (0.0 إلى 1.0)

        Returns:
            صورة RGBA شفافة بلون أساسي ثابت
        """
        layer = Image.new("RGBA", size, (0, 0, 0, 0))

        # إنشاء طبقة لون صلبة + قناة alpha ثابتة
        color_layer = Image.new("RGB", size, base_color)
        alpha_layer = Image.new("L", size, int(255 * opacity))

        # دمج القنوات
        return Image.merge("RGBA", (*color_layer.split(), alpha_layer))

    def create_dna_pulse_repair_layer(
        size: Tuple[int, int],
        positive_pulse: Tuple[int, int, int] = (20, 15, 10),     # لإضافة حيوية
        negative_pulse: Tuple[int, int, int] = (-15, -10, -5),   # لقمع التشوهات
        pulse_steps: int = 5,
        initial_opacity: float = 0.40,
        opacity_decay: float = 0.06,
    ) -> Image.Image:
        """
        إنشاء طبقة إصلاح بنبضات DNA (موجب ثم سالب) لإضافة حيوية وقمع تشوهات

        Args:
            size: أبعاد الصورة
            positive_pulse: قيم RGB الموجبة الأساسية
            negative_pulse: قيم RGB السالبة الأساسية
            pulse_steps: عدد النبضات المتتالية
            initial_opacity: الشفافية الأولية
            opacity_decay: مقدار تناقص الشفافية في كل خطوة

        Returns:
            صورة RGBA تحتوي على نبضات متتالية
        """
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")

        for step in range(pulse_steps):
            alpha = int(255 * (initial_opacity - step * opacity_decay))
            if alpha <= 0:
                break

            # نبضة موجبة (إضافة حيوية)
            pos_color = tuple(int(c * (1 + step * 0.08)) for c in positive_pulse)
            draw.rectangle((0, 0, size[0], size[1]), fill=pos_color + (alpha,))

            # نبضة سالبة (قمع تشوهات) بنصف الشفافية تقريبًا
            neg_color = tuple(int(c * (1 - step * 0.1)) for c in negative_pulse)
            draw.rectangle((0, 0, size[0], size[1]), fill=neg_color + (alpha // 2,))

        return layer

    def create_dna_enhancement_stack(size: Tuple[int, int]) -> Image.Image:
        base = create_dna_base_layer(size, opacity=0.35)
        pulse = create_dna_pulse_repair_layer(size, pulse_steps=5)
        return Image.alpha_composite(base, pulse)

    # ====================== 3. إعادة ترميم الشكل الهندسي ======================
    def repair_geometry_with_net(
            self,
            img: Image.Image,
            mask: Image.Image,
            net: Image.Image,
            prompt: str = "high quality, detailed geometry, realistic structure",
            strength: float = 0.32,
            steps: int = 22,
        ) -> Image.Image:
            """إعادة بناء الشكل الهندسي بدقة عالية"""
            result = self.pipeline(
                prompt=prompt,
                negative_prompt="blurry, deformed, low detail, artifacts",
                image=img,
                mask_image=mask,
                control_image=net,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=7.5,
            ).images[0]

            return result

    # ====================== 4. طبقات لونية مخصصة على أضلاع Net ======================
    def add_dna_colored_layers(
            self,
            net_image: Image.Image,
            mask: Image.Image,
            base_colors: List[Tuple[int, int, int]] = [(255, 80, 80), (80, 255, 80), (80, 80, 255)],  # أحمر - أخضر - أزرق
            blend_mode: Literal["density", "wave"] = "density",
            opacity: float = 0.50,
        ) -> Image.Image:
            """
            دمج ألوان DNA-inspired على أضلاع الشبكة (بدون طفرة، دمج فقط)
            """
            net_arr = np.array(net_image.convert("RGBA"))
            mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

            layer = np.zeros(net_arr.shape, dtype=np.uint8)

            # استخراج أضلاع الشبكة
            gray = cv2.cvtColor(net_arr[..., :3], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 160)

            if blend_mode == "density":
                # دمج حسب كثافة الماسك
                intensity = mask_arr ** 1.4
                for i, color in enumerate(base_colors):
                    weight = intensity * (0.7 - i * 0.18)
                    layer[..., :3] += (np.array(color) * weight[..., None]).astype(np.uint8)

            elif blend_mode == "wave":
                # دمج موجي DNA helix style
                h, w = mask_arr.shape
                wave = np.sin(np.linspace(0, 25 * np.pi, w) + np.arange(h)[:, None] * 0.08)
                wave = (wave + 1) / 2
                wave = np.repeat(wave[..., None], 3, axis=2)
                mixed = np.zeros((h, w, 3))
                for i, color in enumerate(base_colors):
                    mixed += np.array(color) * wave * (1 / len(base_colors))
                layer[..., :3] = mixed.astype(np.uint8)

            layer[..., 3] = (mask_arr * 255 * opacity).astype(np.uint8)
            layer[edges > 0, :3] = (220, 220, 255)  # خطوط Net فاتحة
            layer[edges > 0, 3] = 240

            return Image.fromarray(layer)

    # ====================== 5. إسقاط نبضي DNA-inspired Coloring ======================
    def dna_color_pulse(
        img: Image.Image,
        mask: Image.Image,
        pulse_steps: int = 6,
        hue_std_base: float = 8.0,          # أساس الانحراف العشوائي لـ Hue
        positive_sat_boost: float = 0.28,   # زيادة التشبع (حيوية/طفرة)
        negative_sat_suppress: float = 0.22, # قمع التشبع (تقليل التشوهات)
        factor_decay: float = 0.60,         # معامل تناقص التأثير
    ) -> Image.Image:
        """
        نبض لوني DNA-inspired: طفرة Hue + تعديل Saturation فقط
        (بدون تغيير السطوع Value)
        """
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

        for step in range(pulse_steps):
            factor = 1.0 - (step / pulse_steps) * factor_decay

            # طفرة Hue عشوائية (DNA-like mutation)
            hue_shift = np.random.normal(0, hue_std_base * factor, size=mask_arr.shape) * mask_arr

            # زيادة التشبع (حيوية)
            sat_boost = 1.0 + positive_sat_boost * factor * mask_arr

            # قمع التشبع (توازن)
            sat_suppress = 1.0 - negative_sat_suppress * factor * mask_arr

            # تحويل إلى HSV
            hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

            # تطبيق التغييرات
            hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost * sat_suppress, 0, 255)

            # رجوع إلى RGB
            arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def dna_full_pulse(
        img: Image.Image,
        mask: Image.Image,
        pulse_steps: int = 6,
        hue_std_base: float = 6.0,
        positive_sat_boost: float = 0.25,
        negative_sat_suppress: float = 0.18,
        positive_val_boost: float = 0.15,   # زيادة السطوع (حيوية إضافية)
        factor_decay: float = 0.65,
    ) -> Image.Image:
        """
        نبض DNA-inspired كامل: طفرة Hue + Saturation + Value
        (نسخة أقوى وأكثر حيوية من الدالة السابقة)
        """
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

        for step in range(pulse_steps):
            factor = 1.0 - (step / pulse_steps) * factor_decay

            hue_shift = np.random.normal(0, hue_std_base * factor, size=mask_arr.shape) * mask_arr
            sat_boost = 1.0 + positive_sat_boost * factor * mask_arr
            sat_suppress = 1.0 - negative_sat_suppress * factor * mask_arr
            val_boost = 1.0 + positive_val_boost * factor * mask_arr

            hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

            hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost * sat_suppress, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * val_boost, 0, 255)

            arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # ====================== الدالة الرئيسية (التنفيذ الكامل) ======================
    def repair(
        self,
        img: Image.Image,
        prompt: str = "masterpiece, best quality, highly detailed",
        use_colored_layers: bool = True,
        use_color_pulsing: bool = True,
        pulse_steps: int = 6,
    ) -> Image.Image:
        img = self._ensure_rgb(img)

        # 1. اكتشاف المناطق الميتة / الماسك
        mask = self.detect_dead_zones(img, control_type="tile", threshold=0.35)

        # 2. توليد الشبكة الهيكلية (Net)
        net = self.generate_net_structure(
            img=img,
            mask=mask,
            control_type="union",
            net_strength=0.68,
            steps=18,
        )

        # 3. طبقة DNA أساسية شفافة (اختياري – لو عايز تحتفظ بيها)
        dna_base = self.create_dna_base_layer(img.size, opacity=0.38)

        # 4. الترميم الهندسي الرئيسي باستخدام الشبكة
        repaired = self.repair_geometry_with_net(
            img=img,
            mask=mask,
            net=net,
            prompt=prompt,
            strength=0.35,          # قيمة معتدلة
            steps=25,
        )

        # 5. إضافة طبقات لونية على أضلاع الـ Net (اختياري)
        if use_colored_layers:
            colored = self.add_colored_layers_on_net(
                net_image=net,
                mask=mask,
                colors=[(220, 60, 30), (40, 180, 60), (80, 180, 255)],  # نار - طبيعة - جليد مثلاً
                blend_mode="density",
                base_opacity=0.55,
            )
            repaired = Image.alpha_composite(repaired.convert("RGBA"), colored)

        # 6. نبض لوني DNA-inspired (اختياري)
        if use_color_pulsing:
            final = self.dna_full_pulse(
                img=repaired,
                mask=mask,
                pulse_steps=pulse_steps,
                hue_std_base=7.0,
                positive_sat_boost=0.26,
                negative_sat_suppress=0.20,
                positive_val_boost=0.14,
                factor_decay=0.62,
            )
        else:
            final = repaired

        mask = self.detect_dead_zones(
            img,
            method="canny_dilate",
            canny_low=50,
            canny_high=160,
            dilation_kernel_size=11,     # كبره لو المناطق الميتة كبيرة
            threshold=0.25               # قلله لو عايز تمسك مناطق أكتر
        )

        # 7. تلميع نهائي خفيف
        final = ImageEnhance.Sharpness(final).enhance(1.10)
        final = ImageEnhance.Contrast(final).enhance(1.05)

        return final.convert("RGB")


# ────────────────────────────────────────────────
# استخدام سهل
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. إنشاء محرك الإصلاح
    repair_system = DNANetPulseRepair()

    # 2. فتح الصورة
    try:
        image = Image.open("input.jpg")
    except FileNotFoundError:
        print("خطأ: ملف input.jpg غير موجود")
        exit(1)

    # 3. الإصلاح (لو الدالة موجودة)
    result = repair_system.repair(
        image,
        prompt="highly detailed, realistic skin, vibrant colors",
        use_colored_layers=True,
        use_color_pulsing=True,
        pulse_steps=7
    )

    result.save("dna_net_pulse_repaired.jpg")
    print("✅ تم الترميم بنجاح → dna_net_pulse_repaired.jpg")

    # 4. محرك الألوان
    color_engine = DndSeedColorEngine()   # ← الكائن ده هو الحل

    seed1 = color_engine.generate_dnd_seed_color("Fire")
    seed2 = color_engine.generate_dnd_seed_color("Ice")

    print(f"Seed 1 (Fire): {seed1}  → {DND_COLOR_MENU['Fire']['hex']}")
    print(f"Seed 2 (Ice):  {seed2}  → {DND_COLOR_MENU['Ice']['hex']}")

    # مزج
    mixed = color_engine.mix_dnd_seed_colors(seed1, seed2, ratio=0.45)

    print(f"Mixed color: {mixed}")

    # مراقبة (لو الدالة موجودة)
    report = color_engine.monitor_dnd_color_mix(seed1, seed2, mixed, ratio=0.45)
    print("\nتقرير المزج:")
    for k, v in report.items():
        print(f"  {k}: {v}")
