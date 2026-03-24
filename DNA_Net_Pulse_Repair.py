# DNA_Net_Pulse_Repair.py
"""
DNA_Repair_Pipeline.py

النهج الجديد: DNA-inspired Layer + ControlNet-guided Net
مع إسقاط نبضي DNA-inspired coloring (موجب/سالب) + طبقات لونية مخصصة
"""

from __future__ import annotations
import torch.nn as nn
from typing import Optional, Literal, Tuple, List, Dict, Any, Tuple
import numpy as np

import cv2
import sys
import random
import os
from scipy.ndimage import gaussian_filter

import PIL
from PIL import Image as PILImage
from PIL import Image as ImageOps
from PIL import Image as PILImageFilter
from PIL import ImageEnhance
from PIL import ImageFilter

from union_multi_inpainting import union_img2img_with_mask

# ─── Diffusers & ControlNet ────────────────────────────────────────
from diffusers.pipelines.controlnet.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel

import torch
import torch

print(torch.__version__)
g = torch.Generator(device="cuda")
print(g)

from typing import TypeAlias
TorchDType: TypeAlias = torch.dtype

from typing import TypeAlias
TDTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ثم استخدمه كده:
DTYPE: TorchDType = torch.float16 if torch.cuda.is_available() else torch.float32


# ─── جهاز و dtype مركزيين ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32   # type: ignore[attr-defined]

LOCAL_CONTROLNET_PATH = r"C:\Users\Rashed_Dadou\Desktop\SuperVisorSmartReporter\Rehabilitation Pipeline\update\Update"


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

# ──────────────────────────────────────────────────────────────
#     كلاس class Dnd Seed Color Engine
# ──────────────────────────────────────────────────────────────
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

    def generate_dnd_seed_color(self, element="random", variation=0.12, brightness_boost=0.0) -> Tuple[int, int, int]:
        if element == "random" or element not in self.color_menu:
            element = random.choice(self.elements)

        base = self.color_menu[element]["base_rgb"]

        # حساب مع clip في كل خطوة عشان نتجنب overflow
        r = base[0] * (1 + random.uniform(-variation, variation))
        g = base[1] * (1 + random.uniform(-variation, variation))
        b = base[2] * (1 + random.uniform(-variation, variation))

        # تطبيق brightness بعدين
        r *= (1 + brightness_boost)
        g *= (1 + brightness_boost)
        b *= (1 + brightness_boost)

        # clip نهائي وتحويل لـ int
        color = tuple(int(round(x)) for x in np.clip([r, g, b], 0, 255))

        if any(c < 0 or c > 255 for c in color):
            print(f"تحذير: لون خارج النطاق بعد clip! {color} من {element}")

        print(f"generate_dnd_seed_color → {element:8} → {color}")
        return tuple(np.clip([r, g, b], 0, 255).astype(int))  # هنا النوع

    def generate_palette(
        self,
        count: int = 5,
        base_element: str = "random",
        variation: float = 0.15,
        brightness_range: Tuple[float, float] = (-0.1, 0.25),
    ) -> List[Tuple[int, int, int]]:
        """توليد لوحة ألوان متجانسة بنفس العنصر أو مختلطة"""
        palette = []
        current_element = base_element if base_element != "random" else random.choice(self.elements)

        for _ in range(count):
            boost = random.uniform(*brightness_range)
            col = self.generate_dnd_seed_color(
                element=current_element,
                variation=variation,
                brightness_boost=boost
            )
            palette.append(col)

            # تغيير العنصر أحيانًا لو عايزين تنويع
            if random.random() < 0.25:
                current_element = random.choice(self.elements)

        return palette

    def safe_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """تحويل آمن إلى uint8 مع clip و round"""
        if arr.dtype == np.uint8:
            return arr
        if arr.dtype in (np.float32, np.float64):
            return np.clip(np.round(arr), 0, 255).astype(np.uint8)
        raise TypeError(f"نوع غير مدعوم للتحويل: {arr.dtype}")


    #              نظام توليد الألوان mix_dnd_seed_colors
    # ──────────────────────────────────────────────────────────────
    def mix_dnd_seed_colors(
        self,
        color1: Tuple[int, int, int],
        color2: Optional[Tuple[int, int, int]] = None,
        ratio: float = 0.5,
        element_influence: float = 0.3,
        chaos_factor: float = 0.08,
    ) -> Tuple[int, int, int]:
        """
        مزج لونين بأسلوب DNA-inspired:
          1. مزج خطي أساسي
          2. تأثير طاقة عنصري (اختياري)
          3. طفرة جينية خفيفة (chaos)
        """
        color1 = self._validate_color_tuple(color1, "color1")

        if color2 is None:
            color2 = self.generate_dnd_seed_color("random")
        else:
            color2 = self._validate_color_tuple(color2, "color2")

        # ─── الجزء 1: المزج الخطي الأساسي ─────────────────────────────
        blended = self._linear_blend(color1, color2, ratio)

        # ─── الجزء 2: تأثير الطاقة العنصرية (DNA energy flow) ────────
        if element_influence > 0:
            blended = self._apply_elemental_energy(blended, color1, color2, element_influence)

        # ─── الجزء 3: الطفرة الجينية الخفيفة ───────────────────────────
        final_color = self._apply_genetic_mutation(blended, chaos_factor)

        return final_color

    def _validate_color_tuple(
        self,
        color: Tuple[int, int, int],
        param_name: str = "color"
    ) -> Tuple[int, int, int]:
        if not isinstance(color, tuple) or len(color) != 3:
            raise TypeError(f"{param_name} يجب أن يكون tuple مكون من 3 أعداد")

        try:
            r, g, b = (int(v) for v in color)
        except (ValueError, TypeError):
            raise ValueError(f"لا يمكن تحويل قيم {param_name} إلى أعداد صحيحة: {color}")

        # هنا نستخدم r,g,b بدل col و name
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError(f"قيم {param_name} خارج النطاق 0-255: {color}")

        return (r, g, b)

    def _linear_blend(
        self,
        c1: Tuple[int, int, int],
        c2: Tuple[int, int, int],
        ratio: float,
    ) -> Tuple[int, int, int]:
        """مزج خطي بسيط بين لونين"""
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        return (r, g, b)

    def _apply_elemental_energy(
        self,
        current: Tuple[int, int, int],
        c1: Tuple[int, int, int],
        c2: Tuple[int, int, int],
        influence: float,
    ) -> Tuple[int, int, int]:
        """تطبيق تأثير الطاقة العنصرية بناءً على فرق السطوع"""
        energy_diff = (sum(c1) - sum(c2)) / 765.0  # نطاق تقريبي -1 إلى +1

        r, g, b = current
        r += int(energy_diff * 40 * influence)
        g += int(-energy_diff * 30 * influence)
        b += int(energy_diff * 20 * influence)

        return (r, g, b)

    def _apply_genetic_mutation(
        self,
        color: Tuple[int, int, int],
        chaos_factor: float,
    ) -> Tuple[int, int, int]:
        """إضافة طفرة عشوائية خفيفة (توزيع غاوسي)"""
        deviation = chaos_factor * 30  # القيمة القصوى المتوقعة للطفرة

        r, g, b = color
        r += int(random.gauss(0, deviation))
        g += int(random.gauss(0, deviation))
        b += int(random.gauss(0, deviation))

        # قص القيم للنطاق الصالح
        return tuple(np.clip([r, g, b], 0, 255).astype(int))

    #              نظام دمج الألوان monitor_dnd_color_mix
    # ──────────────────────────────────────────────────────────────
    def monitor_dnd_color_mix(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        result: Tuple[int, int, int],
        ratio: float = 0.5,
    ) -> Dict[str, Any]:
            """
            إنشاء تقرير تحليلي عن عملية مزج الألوان

            Returns:
                قاموس يحتوي على:
                - brightness_balance
                - dominant_element
                - energy_flow
                - mutation_effect
                - original_avg1, original_avg2, result_avg
                - error (في حالة حدوث مشكلة)
            """
            report: Dict[str, Any] = {}

            try:
                # ─── التحقق من صحة الأنواع والقيم ─────────────────────────────
                for name, col in [("color1", color1), ("color2", color2), ("result", result)]:
                    if not isinstance(col, tuple) or len(col) != 3:
                        raise TypeError(f"{name} يجب أن يكون tuple مكون من 3 أعداد")
                    if not all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in col):
                        raise ValueError(f"قيم {name} غير صالحة: {col}")

                # ─── الحسابات الأساسية ────────────────────────────────────────
                avg1 = sum(color1) / 3.0
                avg2 = sum(color2) / 3.0
                avg_res = sum(result) / 3.0

                # تجنب التقسيم على صفر
                expected_avg = (avg1 * (1 - ratio)) + (avg2 * ratio)
                brightness_balance = avg_res / expected_avg if expected_avg > 1e-6 else 0.0

                # ─── تحديد العنصر المهيمن ─────────────────────────────────────
                r, g, b = result
                if r > g + b:
                    dominant = "Fire"
                elif g > r + b:
                    dominant = "Nature"
                elif b > r + g:
                    dominant = "Arcane/Ice"
                else:
                    dominant = "Neutral"

                # ─── تدفق الطاقة ───────────────────────────────────────────────
                avg_original = (avg1 + avg2) / 2
                if avg_res > avg_original + 1.0:
                    energy = "موجب قوي"
                elif avg_res < avg_original - 1.0:
                    energy = "سالب قوي"
                elif abs(avg_res - avg_original) < 1.0:
                    energy = "متوازن"
                else:
                    energy = "سالب متوازن" if avg_res < avg_original else "موجب خفيف"

                # ─── نسبة الطفرة ────────────────────────────────────────────────
                diff = abs(avg_res - expected_avg)
                mutation_pct = (diff * 100) / 255.0
                mutation_str = f"±{int(round(mutation_pct))}%"

                # ─── ملء التقرير ────────────────────────────────────────────────
                report.update({
                    "brightness_balance": round(brightness_balance, 3),
                    "dominant_element": dominant,
                    "energy_flow": energy,
                    "mutation_effect": f"طفرة لونية بنسبة {mutation_str}",
                    "original_avg1": round(avg1, 2),
                    "original_avg2": round(avg2, 2),
                    "result_avg": round(avg_res, 2),
                    "expected_avg": round(expected_avg, 2),
                    "difference": round(diff, 2),
                })

            except Exception as e:
                report["error"] = str(e)
                report["color1"] = color1
                report["color2"] = color2
                report["result"] = result
                report["ratio"] = ratio
                report["status"] = "فشل الحساب"

            return report

# ──────────────────────────────────────────────────────────────
#        كلاس class DNA Net Pulse Repair
# ──────────────────────────────────────────────────────────────
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
        # ====================== 1. إعداد الجهاز والـ dtype ======================
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dtype = dtype if dtype is not None else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        print(f"→ Device: {self.device} | dtype: {self.dtype}")

        # ربط محرك الألوان
        self.color_engine = color_engine
        if self.color_engine:
            print("→ Color engine تم ربطه بنجاح")
        else:
            print("→ No external color engine → using defaults")

        # ====================== 2. تحميل ControlNet (محلي → HF) ======================
        print("\n=== تحميل ControlNet Union ===")
        try:
            self.controlnet = ControlNetUnionModel.from_pretrained(
                LOCAL_CONTROLNET_PATH,
                torch_dtype=self.dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            print("✓ ControlNet Union تم تحميله من المسار المحلي")
        except Exception as e:
            print(f"⚠ فشل التحميل المحلي: {e}")
            print("→ جاري التحميل من Hugging Face...")
            self.controlnet = ControlNetUnionModel.from_pretrained(
                controlnet_model,
                torch_dtype=self.dtype,
                variant=variant,
                use_safetensors=True,
            )
            print("✓ ControlNet Union تم تحميله من HF")

        self.controlnet = self.controlnet.to(self.device)

        # ====================== 3. تحميل الـ Pipeline (نفضل الـ Img2Img variant) ======================
        print("\n=== تحميل Pipeline (Img2Img variant مفضل لدعم strength) ===")

        from diffusers import AutoencoderKL  # للـ vae fix إذا لزم

        try:
            # المحاولة الأولى: Img2Img variant (يدعم strength + image كـ init)
            from diffusers.pipelines.controlnet.pipeline_controlnet_union_sd_xl_img2img import (
                StableDiffusionXLControlNetUnionImg2ImgPipeline
            )

            self.pipeline = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
                sd_model,
                controlnet=self.controlnet,
                vae=AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype
                ),
                torch_dtype=self.dtype,
                variant=variant,
                use_safetensors=True,
                safety_checker=None,
            )
            print("✓ Pipeline Img2Img variant تم تحميله بنجاح (يدعم strength)")

        except (ImportError, Exception) as e:
            print(f"!! فشل تحميل Img2Img variant: {e}")
            print("→ fallback إلى النسخة txt2img الأساسية + الاعتماد على الدوال المنفصلة للـ img2img")

            self.pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
                sd_model,
                controlnet=self.controlnet,
                vae=AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype
                ),
                torch_dtype=self.dtype,
                variant=variant,
                use_safetensors=True,
                safety_checker=None,
            )
            print("✓ Pipeline (fallback txt2img) تم تحميله")

        # ====================== 4. نقل + تفعيل التحسينات (بالترتيب الصحيح) ======================
        self.pipeline = self.pipeline.to(self.device)

        if enable_attention_slicing:
            self.pipeline.enable_attention_slicing("max")
            print("→ Attention slicing مفعّل")

        if enable_cpu_offload and self.device.type == "cuda":
            try:
                self.pipeline.enable_model_cpu_offload()
                print("→ Model CPU offload مفعّل (موفر VRAM)")
            except Exception as e:
                print(f"!! فشل CPU offload: {e}")
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    print("→ Sequential CPU offload مفعّل بدلاً منه")
                except Exception as e2:
                    print(f"!! فشل sequential offload أيضًا: {e2}")

        # تنظيف الذاكرة
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # ====================== 5. تقرير نهائي ======================
        print("\n" + "═" * 75)
        print("✅ DNANetPulseRepair تم تهيئته بنجاح")
        print("═" * 75)
        print(f"   • Device          : {self.device}")
        print(f"   • Dtype           : {self.dtype}")
        print(f"   • Pipeline type   : {'Img2Img variant' if 'Img2Img' in str(type(self.pipeline).__name__) else 'Txt2Img fallback'}")
        print(f"   • Offload         : {'مفعّل' if enable_cpu_offload and self.device.type=='cuda' else 'غير مفعّل'}")
        if self.device.type == "cuda":
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   • VRAM reserved   : {reserved:.2f} / {total:.2f} GiB")
        print("═" * 75)

    def _ensure_rgb(self, img: PILImage.Image) -> PILImage.Image:
        """
        التأكد من أن الصورة في وضع RGB
        """
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    # ──────────────────────────────────────────────────────────────
    #                دوال الكشف والتحضير (Helper)
    # ──────────────────────────────────────────────────────────────
    def detect_dead_zones(
        self,
        img: PILImage.Image,
        method: Literal["multi", "canny_dilate", "laplacian", "entropy"] = "multi",
        threshold: float = 0.28,
        canny_low: int = 60,
        canny_high: int = 180,
        dilation_kernel_size: int = 9,
        min_area_ratio: float = 0.008,
        return_type: Literal["mask", "score_map", "signed_map"] = "mask",
        debug: bool = False,
    ) -> PILImage.Image:
        """
        كشف المناطق الميتة أو منخفضة التفاصيل (Dead Zones)
        """
        if debug:
            print("\n" + "="*65)
            print(f"بدء detect_dead_zones | method={method} | return={return_type}")
            print("="*65)

        # ====================== 1. التحضير ======================
        img_np = np.array(img.convert("RGB"))
        h, w = img_np.shape[:2]               # ← هنا التصحيح المهم جدًا
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # ====================== 2. حساب المقاييس الأساسية ======================
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = gaussian_filter(np.abs(lap), sigma=1.0)
        contrast_score = lap_var / (lap_var.max() + 1e-8)

        edges = cv2.Canny(gray.astype(np.uint8), canny_low, canny_high)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (7, 7), 0)

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_score = hsv[..., 1] / 255.0
        val_score = hsv[..., 2] / 255.0

        # ====================== 3. حساب الـ Final Score حسب الطريقة ======================
        if method == "multi":
            final_score = (
                0.40 * (1.0 - contrast_score) +
                0.30 * (1.0 - edge_density) +
                0.20 * (1.0 - sat_score) +
                0.10 * (val_score < 0.08).astype(np.float32)
            )

        elif method == "canny_dilate":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            final_score = 1.0 - (dilated.astype(np.float32) / 255.0)

        elif method == "laplacian":
            final_score = 1.0 - np.clip(contrast_score, 0.0, 1.0)

        elif method == "entropy":
            from scipy.stats import entropy
            entropy_map = np.zeros_like(gray)

            patch_size = 32
            step = 16

            # نستخدم int صريحًا لتجنب مشاكل float
            for i in range(0, int(h), step):
                for j in range(0, int(w), step):
                    patch = gray[i:i+patch_size, j:j+patch_size].ravel()
                    if len(patch) > 0:
                        hist, _ = np.histogram(patch, bins=64, range=(0, 255), density=True)
                        ent = entropy(hist + 1e-8)
                        entropy_map[i:i+patch_size, j:j+patch_size] = ent

            final_score = 1.0 - (entropy_map / np.log(64))

        else:
            raise ValueError(f"method غير مدعوم: {method}")

        # ====================== 4. تنظيف وتحسين الـ Score ======================
        final_score = cv2.GaussianBlur(final_score, (5, 5), 1.0)
        final_score = (final_score - final_score.mean()) / (final_score.std() + 1e-8)
        final_score = np.clip(final_score, -1.8, 1.8)

        # ====================== 5. إنشاء المخرج حسب return_type ======================
        if return_type == "mask":
            dead_mask = (final_score < -threshold).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dead_mask = cv2.morphologyEx(dead_mask, cv2.MORPH_OPEN, kernel)
            dead_mask = cv2.dilate(dead_mask, np.ones((5, 5), np.uint8), iterations=1)

            result = PILImage.fromarray(dead_mask).convert("L")

        elif return_type == "score_map":
            vis = np.clip((final_score + 1.8) / 3.6 * 255, 0, 255).astype(np.uint8)
            result = PILImage.fromarray(vis).convert("L")

        elif return_type == "signed_map":
            score_norm = np.clip(final_score / 2.0, -1.0, 1.0)
            hue = np.where(score_norm <= 0,
                        0 + (score_norm + 1) * 30,
                        30 + score_norm * 55).astype(np.float32)

            hsv = np.stack([hue, np.ones_like(hue) * 255, np.ones_like(hue) * 255], axis=-1).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            result = PILImage.fromarray(rgb)

        if debug:
            print(f"→ تم إنشاء dead zones بنجاح | method={method} | threshold={threshold}")
            print("="*65 + "\n")

        # حماية نهائية: لو النتيجة None (مستحيل بعد التصحيح لكن احتياطي)
        if result is None:
            result = PILImage.new("L", img.size, 0)

        return result

    def _create_canny_control(
        self,
        img_np: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
        blur_kernel_size: int = 5,
        dilate_kernel_size: int = 3,
        dilate_iterations: int = 1,
    ) -> PILImage.Image:
        """
        إنشاء صورة تحكم بنمط Canny مع معالجة مسبقة وتوسيع للخطوط
        (دالة مساعدة داخلية)
        """
        # 1. التعامل مع أنواع الصور المختلفة (RGB, RGBA, Grayscale)
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:        # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            elif img_np.shape[2] == 3:      # RGB
                pass
            else:
                raise ValueError(f"عدد القنوات غير مدعوم: {img_np.shape[2]}")
        elif len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"شكل الصورة غير مدعوم: {img_np.shape}")

        # 2. تحويل إلى Grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 3. تقليل الضوضاء بـ Gaussian Blur
        if blur_kernel_size > 1:
            gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

        # 4. تطبيق Canny Edge Detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # 5. توسيع الخطوط (Dilation) لربط الخطوط المتقطعة
        if dilate_kernel_size > 1:
            kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

        # 6. تحويل النتيجة إلى RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return PILImage.fromarray(edges_rgb)

    def _create_lineart_control(
        self,
        img_np: np.ndarray,
        thickness: int = 11,
        block_size: Optional[int] = None,
    ) -> PILImage.Image:
        """
        إنشاء صورة تحكم بنمط Line Art باستخدام Adaptive Threshold
        (دالة مساعدة داخلية)
        """
        # 1. التعامل مع الصورة (RGB أو RGBA أو Grayscale)
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:   # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            # RGB يبقى كما هو
        elif len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"شكل الصورة غير مدعوم: {img_np.shape}")

        # 2. تحويل إلى Grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 3. حساب حجم البلوك ديناميكياً
        if block_size is None:
            block_size = 11 + thickness * 2

        # 4. تطبيق Adaptive Threshold لإنشاء Line Art
        lineart = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=2
        )

        # 5. عكس الألوان (الخطوط تصبح بيضاء على خلفية سوداء)
        lineart = 255 - lineart

        return PILImage.fromarray(lineart).convert("RGB")

    # ──────────────────────────────────────────────────────────────
    #                دوال التوليد والإصلاح الرئيسية
    # ──────────────────────────────────────────────────────────────
    NET_POSITIVE_PROMPT = (
        "structural grid, clean architectural lines, "
        "technical blueprint style, high contrast edges, "
        "precise geometric network, technical drawing, "
        "sharp vector lines, schematic, diagram"
    )

    NET_NEGATIVE_PROMPT = (
        "blurry, noisy, low detail, artifacts, text, watermark, "
        "overexposed, underexposed, deformed, low quality, "
        "bad anatomy, jpeg artifacts, compression, grainy"
    )

    def generate_net_structure(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        control_type: Literal["union", "canny", "lineart"] = "union",
        net_strength: float = 0.68,
        steps: int = 18,
        guidance_scale: float = 7.2,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        debug: bool = False,
        seed: Optional[int] = 42,
    ) -> PILImage.Image:
        """
        توليد الشبكة الهيكلية (Structural Net) باستخدام ControlNet Union
        """
        if debug:
            print("\n" + "═"*70)
            print("بدء generate_net_structure")
            print("═"*70)
            print(f"  img size  : {img.size if img else 'None'}")
            print(f"  mask size : {mask.size if mask else 'None'}")

        # ─── 1. التحضير والحماية من None ───────────────────────────────────────
        if img is None or mask is None:
            print("!! img أو mask = None → إرجاع الصورة الأصلية كـ fallback")
            return img if img is not None else PILImage.new("RGB", (512, 512), (0,0,0))

        img = img.convert("RGB")
        mask = mask.convert("L")

        # تحقق الحجم قبل أي عملية
        if img.size != mask.size:
            print(f"تحذير: أحجام مختلفة → img {img.size} vs mask {mask.size} → تصغير الماسك")
            mask = mask.resize(img.size, PILImage.NEAREST)

        if prompt is None:
            prompt = self.NET_POSITIVE_PROMPT or "detailed structure, clean lines, high quality net, technical blueprint"
        if negative_prompt is None:
            negative_prompt = self.NET_NEGATIVE_PROMPT or "blurry, noisy, artifacts, deformed"

        # ─── 2. إعداد صورة التحكم ────────────────────────────────────────────────
        control_img = img  # default

        try:
            if control_type == "canny":
                control_img = self._create_canny_control(np.array(img), low_threshold=60, high_threshold=180)
            elif control_type == "lineart":
                control_img = self._create_lineart_control(np.array(img), thickness=11)
            elif control_type == "union":
                if debug:
                    print("→ استخدام control_type = union (الصورة الأصلية مباشرة)")
        except Exception as e:
            print(f"خطأ في إنشاء control image ({control_type}): {e}")
            control_img = img  # fallback

        if control_img.size != img.size:
            control_img = control_img.resize(img.size, PILImage.LANCZOS)

        # ─── 3. الاستدلال ─────────────────────────────────────────────────────────
        try:
            # تأكيد القيم الرقمية
            net_strength   = float(net_strength)   if net_strength is not None else 0.68
            steps          = int(steps)            if steps is not None else 18
            guidance_scale = float(guidance_scale) if guidance_scale is not None else 7.2

            generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

            if debug:
                print(f"→ استدعاء pipeline | strength={net_strength:.2f}, steps={steps}, guidance={guidance_scale:.1f}")

            with torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    control_image=control_img,
                    controlnet_conditioning_scale=net_strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            # استخراج النتيجة بأمان
            if hasattr(output, 'images') and output.images:
                result = output.images[0]
            elif isinstance(output, (list, tuple)) and output:
                result = output[0]
            elif isinstance(output, PILImage.Image):
                result = output
            else:
                print("⚠ نوع الـ output غير متوقع:", type(output))
                result = img

            if debug:
                print("✅ تم توليد الشبكة الهيكلية")

        except Exception as e:
            print(f"!! خطأ في generate_net_structure: {type(e).__name__}: {str(e)}")
            result = img

        if debug:
            print("═"*70 + "\n")

        return result


    # ====================== إعادة ترميم الشكل الهندسي ======================
    def repair_geometry_with_net(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        net: PILImage.Image,
        prompt: str = "masterpiece, best quality, highly detailed, realistic skin texture",
        negative_prompt: Optional[str] = None,
        strength: float = 0.35,
        steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.80,
        control_type: Optional[int] = None,  # إذا كان Union يدعم تحديد نوع الـ control
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> PILImage.Image:
        """
        الإصلاح الهندسي الرئيسي باستخدام الشبكة (net) كـ control image
        تستخدم img2img + inpainting + ControlNet Union

        Args:
            strength: قوة الـ denoising (0.25–0.45 عادةً كافية للإصلاح الخفيف)
            steps: عدد خطوات الـ inference (20–35 جيد)
            controlnet_conditioning_scale: قوة تأثير الـ control image (0.6–1.0)
        """
        if debug:
            print("\n" + "="*75)
            print("بدء repair_geometry_with_net")
            print(f"strength={strength:.2f} | steps={steps} | control_scale={controlnet_conditioning_scale:.2f}")
            print("="*75)

        # ─── 1. التحضير والتحقق من الأحجام ──────────────────────────────────────
        img   = img.convert("RGB")
        mask  = mask.convert("L")
        net   = net.convert("RGB")

        if not (img.size == mask.size == net.size):
            raise ValueError(
                f"أحجام غير متطابقة → img: {img.size}, mask: {mask.size}, net: {net.size}"
            )

        # negative prompt افتراضي إذا لم يُمرر
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, deformed, bad anatomy, extra limbs, "
                "poorly drawn face, bad proportions, watermark, text, signature"
            )

        # ─── 2. إعداد الـ kwargs للـ pipeline ──────────────────────────────────────
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            # "image": img,             ← احذفه أو علّقه
            # "mask_image": mask,       ← احذفه أو علّقه
            "control_image": net,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }

        if control_type is not None:
            pipe_kwargs["control_type"] = control_type

        if seed is not None:
            pipe_kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipeline(**pipe_kwargs)
        repaired = output.images[0]

        # ─── 3. التنفيذ مع معالجة الأخطاء ────────────────────────────────────────
        try:
            with torch.inference_mode():
                output = self.pipeline(**pipe_kwargs)

            repaired = output.images[0]

            if debug:
                print("→ الإصلاح الهندسي تم بنجاح")
                print("="*75 + "\n")

            return repaired.convert("RGB")

        except Exception as e:
            print("\n!! خطأ في repair_geometry_with_net:")
            print(f"   {type(e).__name__}: {str(e)}")
            print("→ إرجاع الصورة الأصلية كـ fallback\n")
            return img

    def repair_with_pulse_layer(
        self,
        img: PILImage.Image,
        prompt: str = "high quality, realistic details, vibrant colors",
        control_type: Literal["tile", "canny", "depth"] = "tile",
        pulse_steps: int = 5,
        blend_opacity: float = 0.65,
    ) -> PILImage.Image:
        """
        الدالة الرئيسية للترميم الطبقي النبضي
        """
        img = self._ensure_rgb(img)

        # 1. اكتشاف المناطق المنهارة (اختياري خارجي لو عايز)
        # mask = self.detect_dead_zones(img, control_type=control_type)  # ← ممكن تشيله لو الداخلي كفاية

        # 2. إنشاء الطبقة النبضية + الإصلاح المباشر
        repaired = self.create_dna_pulse_repair_layer(
            img=img,                        # ← مرر الصورة عشان تعمل detect داخلها
            pulse_steps=pulse_steps,
            initial_opacity=0.45,
            opacity_decay=0.05,
            blur_radius=1,
            debug_save=True,                # لو عايز تشوف signed_map
        )

        # 3. تلميع نهائي (اختياري)
        repaired = ImageEnhance.Sharpness(repaired).enhance(1.12)
        repaired = ImageEnhance.Contrast(repaired).enhance(1.06)

        return repaired.convert("RGB")

    def _prepare_control_image(
        self,
        img: PILImage.Image,
        control_type: str,
        canny_low: int,
        canny_high: int,
        lineart_thickness: int,
    ) -> PILImage.Image:
        """إعداد صورة التحكم (control_image) حسب نوع الـ ControlNet المطلوب"""
        img_np = np.array(img)

        if control_type == "canny":
            return self._create_canny_control(img_np, canny_low, canny_high)

        elif control_type == "lineart":
            return self._create_lineart_control(img_np, lineart_thickness)

        elif control_type == "union":
            return img  # Union يستخدم الصورة الأصلية مباشرة

        else:
            supported = ["canny", "lineart", "union"]
            raise ValueError(
                f"نوع التحكم غير مدعوم: {control_type!r}\n"
                f"الأنواع المدعومة حالياً: {', '.join(supported)}"
            )

    def get_net_positive_prompt(self) -> str:
        return self.NET_POSITIVE_PROMPT

    def get_net_negative_prompt(self) -> str:
        return self.NET_NEGATIVE_PROMPT

    def _run_controlnet_inference(
        self,
        image: PILImage.Image,              # هتستخدمها لاحقًا في الدمج فقط
        mask_image: PILImage.Image,         # هتستخدمها لاحقًا في الدمج فقط
        control_image: PILImage.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance_scale: float,
    ) -> PILImage.Image:
        """
        تنفيذ inference باستخدام Union Pipeline مباشرة
        (بدون image/strength لأن Union لا يدعمهما في __call__)
        """
        print(f"→ _run_controlnet_inference | steps={steps} | guidance={guidance_scale:.1f}")

        try:
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,                     # ← الـ control الوحيد المدعوم
                controlnet_conditioning_scale=strength,          # ← نستخدم strength هنا كـ scale
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator = torch.Generator(device=self.device).manual_seed(42) if seed is not None else None,
            )

            generated = output.images[0]
            print("  ✓ تم الـ inference بنجاح (text-to-image + control)")

            # دمج خفيف مع الصورة الأصلية عشان نحافظ على بعض التفاصيل
            # (اختياري - يمكنك تعديل alpha حسب اللي يناسبك)
            alpha = 0.35  # 35% من الجديد + 65% من الأصلي
            blended = Image.blend(image.convert("RGB"), generated.convert("RGB"), alpha)

            return blended

        except Exception as e:
            print(f"  !! خطأ في _run_controlnet_inference: {type(e).__name__}: {e}")
            return image  # fallback

    # ====================== 2. DNA Layer شفافة أولى ======================
    def create_dna_base_layer(
            self,
            size: Tuple[int, int],
            base_color: Tuple[int, int, int] = (40, 120, 60),   # أخضر DNA خفيف
            opacity: float = 0.38,
        ) -> PILImage.Image:
        """
        إنشاء طبقة أساس DNA شفافة ثابتة (تُستخدم كخلفية/أساس للإحياء)

        Args:
            size: أبعاد الصورة (width, height)
            base_color: لون أساسي RGB
            opacity: درجة الشفافية (0.0 إلى 1.0)

        Returns:
            صورة RGBA شفافة بلون أساسي ثابت
        """
        layer = PILImage.new("RGBA", size, (0, 0, 0, 0))

        # إنشاء طبقة لون صلبة + قناة alpha ثابتة
        color_layer = PILImage.new("RGB", size, base_color)
        alpha_layer = PILImage.new("L", size, int(255 * opacity))

        # دمج القنوات
        return Image.merge("RGBA", (*color_layer.split(), alpha_layer))

    def create_dna_pulse_repair_layer(
        self,
        img: PILImage.Image,
        pulse_steps: int = 6,
        element: str = "random",                     # ← جديد: نختار العنصر هنا
        initial_opacity: float = 0.45,
        opacity_decay: float = 0.05,
        blur_radius: int = 1,
        debug_save: bool = False,
    ) -> PILImage.Image:
        """
        طبقة نبضية DNA-inspired مع ربط حقيقي بالـ D&D elements عبر color_engine
        """
        from PIL import Image, ImageDraw, ImageFilter

        size = img.size
        layer = PILImage.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # 1. كشف المناطق المنهارة
        dead_mask = self.detect_dead_zones(img, method="multi", threshold=0.25, return_type="mask")

        # 2. توليد الألوان من color_engine (إذا موجود)
        if self.color_engine:
            # اختيار عنصر إذا كان random
            selected_element = element if element != "random" else random.choice(self.color_engine.elements)

            # لون أساسي موجب (positive pulse)
            base_positive = self.color_engine.generate_dnd_seed_color(
                element=selected_element,
                variation=0.18,
                brightness_boost=0.15
            )

            # لون سالب (negative pulse) → نسخة أغمق / أقل سطوعًا
            base_negative = tuple(max(0, int(c * 0.65 - 20)) for c in base_positive)

            print(f"→ DNA Pulse يستخدم عنصر: {selected_element}")
            print(f"   Positive base: {base_positive}")
            print(f"   Negative base: {base_negative}")
        else:
            # fallback إذا ما فيش engine
            base_positive = (30, 25, 20)
            base_negative = (-10, -8, -5)
            print("→ استخدام ألوان DNA افتراضية (no color engine)")

        # 3. بناء الطبقة النبضية
        for step in range(pulse_steps):
            alpha = int(255 * (initial_opacity - step * opacity_decay))
            if alpha <= 0:
                break

            # موجبة (تزداد قوة نسبيًا مع الخطوات)
            pos_r = int(base_positive[0] * (1 + step * 0.08))
            pos_g = int(base_positive[1] * (1 + step * 0.08))
            pos_b = int(base_positive[2] * (1 + step * 0.08))
            pos_fill = (pos_r, pos_g, pos_b, alpha)
            draw.rectangle((0, 0, size[0], size[1]), fill=pos_fill)

            # سالبة (تقل قوة مع الخطوات)
            neg_r = int(base_negative[0] * (1 - step * 0.12))
            neg_g = int(base_negative[1] * (1 - step * 0.12))
            neg_b = int(base_negative[2] * (1 - step * 0.12))
            neg_fill = (neg_r, neg_g, neg_b, alpha // 2)
            draw.rectangle((0, 0, size[0], size[1]), fill=neg_fill)

        # 4. بلور اختياري + دمج على المناطق السالبة فقط
        if blur_radius > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(blur_radius))

        repaired = PILImage.composite(layer, img.convert("RGBA"), dead_mask.convert("L"))

        return repaired

    # ====================== 4. طبقات لونية مخصصة على أضلاع Net ======================
    def add_dna_colored_layers(
        self,
        net_image: PILImage.Image,
        mask: PILImage.Image,
        base_colors: Optional[List[Tuple[int, int, int]]] = None,
        blend_mode: Literal["dna_gradient", "strand", "balanced", "edge_glow"] = "dna_gradient",
        opacity: float = 0.45,
        edge_boost: float = 0.85,
        use_color_engine: bool = True,
        debug: bool = False,
    ) -> PILImage.Image:
        """
        إضافة طبقات لونية DNA-inspired (الإصدار النقي - مهمتها الأساسية فقط)

        الدالة مسؤولة فقط عن:
        - توليد ألوان DNA أنيقة (موجب/سالب أو متعددة العناصر)
        - رسمها على طبقة RGBA حسب الماسك والشبكة
        - إرجاع الطبقة الملونة فقط (لا نبض ولا طفرة داخلية)
        """
        if debug:
            print("\n" + "="*70)
            print("بدء add_dna_colored_layers - النسخة النقية")
            print("="*70)

        # ====================== 1. التحضير ======================
        h, w = mask.size
        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        mask_arr = np.expand_dims(mask_arr, axis=-1)

        # توليد الألوان باستخدام Color Engine
        if base_colors is None:
            if self.color_engine and use_color_engine:
                elem = random.choice(self.color_engine.elements)
                pos = self.color_engine.generate_dnd_seed_color(elem, variation=0.15, brightness_boost=0.20)
                neg = tuple(max(0, int(c * 0.52 - 28)) for c in pos)
                base_colors = [pos, neg]
                if debug:
                    print(f"→ DNA Element: {elem} | Positive: {pos} | Negative: {neg}")
            else:
                base_colors = [(55, 195, 85), (175, 55, 125)]  # أخضر DNA كلاسيكي وأرجواني

        pos_color = np.array(base_colors[0], dtype=np.float32)
        neg_color = np.array(base_colors[1] if len(base_colors) > 1 else base_colors[0], dtype=np.float32)

        # استخراج الحواف من net_image لتعزيز الشبكة
        net_arr = np.array(net_image.convert("RGB"))
        gray = cv2.cvtColor(net_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 45, 135).astype(np.float32) / 255.0

        # ====================== 2. إنشاء الطبقة اللونية (DNA Style) ======================
        layer = np.zeros((h, w, 4), dtype=np.float32)

        if blend_mode == "dna_gradient":
            # تدرج DNA أنيق (من موجب إلى سالب)
            gradient = np.linspace(0.0, 1.0, w)
            gradient = np.tile(gradient, (h, 1))
            color_map = gradient[..., None] * pos_color + (1.0 - gradient[..., None]) * neg_color
            layer[..., :3] = color_map * mask_arr[..., 0, None]

        elif blend_mode == "strand":
            # نمط خيوط DNA متوازية (أنيق وبيولوجي)
            y = np.linspace(0, 1, h)[:, None]
            strand = np.sin(y * 12) * 0.5 + 0.5
            strand = np.tile(strand, (1, w))
            color_map = strand[..., None] * pos_color + (1.0 - strand[..., None]) * neg_color
            layer[..., :3] = color_map * mask_arr[..., 0, None]

        elif blend_mode == "balanced":
            # توازن متساوي بين الألوان مع تأثير خفيف
            ratio = 0.52
            color_map = ratio * pos_color + (1.0 - ratio) * neg_color
            layer[..., :3] = color_map * mask_arr[..., 0, None]

        elif blend_mode == "edge_glow":
            # تركيز اللون على حواف الشبكة (الأضلاع)
            intensity = np.clip(mask_arr[..., 0] ** 1.35, 0.0, 1.0)
            color_map = 0.65 * pos_color + 0.35 * neg_color
            layer[..., :3] = color_map * intensity[..., None]

            # إضاءة إضافية على الحواف
            layer[..., :3] += (edges[..., None] * edge_boost * 55)

        else:
            raise ValueError(f"blend_mode غير مدعوم: {blend_mode}")

        # ====================== 3. تطبيق Alpha + تعزيز الحواف ======================
        alpha = mask_arr[..., 0] * 255 * opacity
        alpha = np.clip(alpha + edges * 95, 0, 255)   # الحواف تكون أكثر وضوحاً

        layer[..., 3] = alpha
        layer[..., :3] = np.clip(layer[..., :3], 0, 255)

        # ====================== 4. التحويل النهائي ======================
        layer = np.round(layer).astype(np.uint8)
        dna_layer = PILImage.fromarray(layer, mode="RGBA")

        if debug:
            print(f"→ Blend Mode : {blend_mode}")
            print(f"→ Opacity    : {opacity:.2f} | Edge Boost : {edge_boost:.2f}")
            print("✅ تم إنشاء طبقة DNA Colored Layers بنجاح (نسخة نقية)")
            print("="*70 + "\n")

        return dna_layer

    # ====================== 5. إسقاط نبضي DNA-inspired Coloring ======================
    def dna_color_pulse(
        self,
        img: Image.Image,
        mask: Image.Image,
        pulse_steps: int = 6,
        hue_std_base: float = 8.0,          # أساس الانحراف العشوائي لـ Hue
        positive_sat_boost: float = 0.28,   # زيادة التشبع (حيوية/طفرة)
        negative_sat_suppress: float = 0.22, # قمع التشبع (تقليل التشوهات)
        factor_decay: float = 0.60,         # معامل تناقص التأثير
    ) -> PILImage.Image:
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
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        pulse_steps: int = 6,
        hue_std_base: float = 7.2,
        sat_boost: float = 0.27,
        sat_suppress: float = 0.19,
        val_boost: float = 0.15,
        factor_decay: float = 0.63,
        enable_hue: bool = True,
        enable_sat: bool = True,
        enable_val: bool = True,
        clip_hue: bool = True,
        debug: bool = True,
    ) -> PILImage.Image:
        """
        نبض لوني DNA-inspired كامل ومحسن (الإصدار 2.0)

        Args:
            img: الصورة الأصلية (RGB)
            mask: الماسك الذي يحدد المناطق المستهدفة
            pulse_steps: عدد خطوات النبض
            hue_std_base: قوة الطفرة في Hue
            sat_boost: زيادة التشبع في المناطق الحية
            sat_suppress: تقليل التشبع في المناطق الضعيفة
            val_boost: زيادة السطوع
            factor_decay: معامل تناقص التأثير مع الخطوات
            enable_hue/sat/val: تفعيل أو تعطيل كل خاصية على حدة
            clip_hue: تثبيت Hue بين 0-180
            debug: إظهار التقارير التفصيلية
        """
        if debug:
            print("\n" + "="*70)
            print("🚀 بدء dna_full_pulse v2.0 - النبض اللوني DNA")
            print("="*70)

        # تحضير الصورة والماسك
        rgb = np.array(img.convert("RGB"), dtype=np.float32)
        mask_arr = np.expand_dims(np.array(mask.convert("L"), dtype=np.float32) / 255.0, axis=-1)

        # تحويل واحد إلى HSV
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        if debug:
            print(f"  الصورة: {rgb.shape} | pulse_steps: {pulse_steps} | decay: {factor_decay:.3f}")

        # ====================== حلقة النبض ======================
        for step in range(pulse_steps):
            factor = max(0.0, 1.0 - (step / pulse_steps) * factor_decay)

            if debug:
                print(f"\n  Step {step+1:2d}/{pulse_steps} | factor = {factor:.4f}")

            # --- Hue Shift ---
            if enable_hue:
                hue_shift = np.random.normal(0, hue_std_base * factor, size=hsv.shape[:2])
                hsv[..., 0] += hue_shift * mask_arr[..., 0]

                if clip_hue:
                    hsv[..., 0] = np.mod(hsv[..., 0], 180.0)

                if debug:
                    print(f"    Hue  → min: {hsv[...,0].min():5.1f} | max: {hsv[...,0].max():5.1f}")

            # --- Saturation ---
            if enable_sat:
                sat_mult = 1.0 + (sat_boost - sat_suppress) * factor * mask_arr[..., 0]
                sat_mult = np.clip(sat_mult, 0.12, 2.9)
                hsv[..., 1] *= sat_mult
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

                if debug:
                    print(f"    Sat  → mean: {hsv[...,1].mean():5.1f}")

            # --- Value ---
            if enable_val:
                val_mult = 1.0 + val_boost * factor * mask_arr[..., 0]
                val_mult = np.clip(val_mult, 0.65, 1.50)
                hsv[..., 2] *= val_mult
                hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

                if debug:
                    print(f"    Val  → mean: {hsv[...,2].mean():5.1f}")

        # التحويل النهائي
        rgb_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)

        result = PILImage.fromarray(rgb_out)

        if debug:
            print("\n✅ انتهى dna_full_pulse بنجاح (v2.0)")
            print("="*70 + "\n")

        return result

    # ====================== الدالة الرئيسية (التنفيذ الكامل) ======================
    def repair(
        self,
        img: PILImage.Image,
        prompt: str = "masterpiece, best quality, highly detailed, realistic",
        use_colored_layers: bool = True,
        use_color_pulsing: bool = True,
        pulse_steps: int = 6,
        net_strength: float = 0.68,
        repair_strength: float = 0.35,
        zoom_enabled: bool = False,
        zoom_factor: float = 1.5,
        zoom_strength: float = 0.32,
    ) -> PILImage.Image:
        """
        الدالة الرئيسية للترميم الكامل باستخدام DNA-Net Pulse Repair
        تم تنظيفها بالكامل وإزالة التكرار
        """
        img = self._ensure_rgb(img)
        original_size = img.size

        print("\n" + "═" * 70)
        print("🚀 بدء عملية الترميم DNA-Net Pulse Repair")
        print("═" * 70)

        # ====================== 1. كشف المناطق التالفة ======================
        print("→ Step 1: كشف المناطق الميتة...")
        mask = self.detect_dead_zones(
            img=img,
            method="multi",
            threshold=0.32,
            return_type="mask"
        )

        # ──────────────── حماية جديدة مهمة جدًا ────────────────────────
        if mask is None:
            print("⚠️ ماسك None من detect_dead_zones → fallback: استخدام ماسك فارغ (لا مناطق تالفة)")
            mask = PILImage.new("L", img.size, 0)  # ماسك أسود كامل = لا تغيير

        # تحقق إذا كان الماسك فارغ تمامًا (كله أسود)
        mask_array = np.array(mask)
        if mask_array.max() == 0:
            print("⚠️ ماسك فارغ تمامًا (لا مناطق تالفة) → تخطي الترميم وإرجاع الصورة الأصلية")
            return img

        white_pixels = np.sum(mask_array > 0)
        print(f"  → عدد البكسلات البيضاء في الماسك: {white_pixels}")
        if white_pixels < 100:  # حد أدنى صغير جدًا (عدد بكسلات قليل جدًا)
            print("⚠️ مناطق تالفة قليلة جدًا (<100 بكسل) → تخطي الترميم")
            return img
        # ────────────────────────────────────────────────────────────────

        # ====================== 2. توليد الشبكة الهيكلية (Net) ======================
        print("→ Step 2: توليد الشبكة الهيكلية (Structural Net)...")
        net = self.generate_net_structure(
            img=img,
            mask=mask,
            control_type="union",
            net_strength=net_strength,
            steps=18,
            prompt=self.NET_POSITIVE_PROMPT,
            negative_prompt=self.NET_NEGATIVE_PROMPT,
        )

        # ====================== 3. الترميم الهندسي الأساسي ======================
        print("→ Step 3: الترميم الهندسي باستخدام ControlNet...")
        repaired = self.repair_geometry_with_net(
            img=img,
            mask=mask,
            net=net,
            prompt=prompt,
            strength=repair_strength,
            steps=25,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.80,
        )

        # ====================== 4. إضافة الطبقات اللونية DNA ======================
        if use_colored_layers:
            print("→ Step 4: إضافة الطبقات اللونية DNA-inspired...")
            colored_layer = self.add_dna_colored_layers(
                net_image=net,
                mask=mask,
                blend_mode="density",
                opacity=0.52,
                use_random_colors=True
            )
            # دمج الطبقة اللونية
            repaired = Image.alpha_composite(
                repaired.convert("RGBA"),
                colored_layer
            ).convert("RGB")

        # ====================== 5. النبض اللوني الكامل (DNA Pulse) ======================
        if use_color_pulsing:
            print("→ Step 5: تطبيق النبض اللوني DNA Full Pulse...")
            repaired = self.dna_full_pulse(
                img=repaired,
                mask=mask,
                pulse_steps=pulse_steps,
                hue_std_base=7.2,
                positive_sat_boost=0.27,
                negative_sat_suppress=0.19,
                positive_val_boost=0.15,
                factor_decay=0.63,
            )

        # ====================== 6. DNA Zoom Repair (اختياري - للتفاصيل العالية) ======================
        if zoom_enabled:
            print("→ Step 6: تفعيل DNA Zoom Repair لتحسين التفاصيل...")
            repaired = self.dna_zoom_repair(
                img=repaired,
                mask=mask,
                net=net,
                zoom_factor=zoom_factor,
                strength=zoom_strength,
                steps=28,
                controlnet_conditioning_scale=0.85,
            )

        # ====================== 7. التلميع النهائي ======================
        print("→ Step 7: التلميع النهائي...")
        repaired = ImageEnhance.Sharpness(repaired).enhance(1.12)
        repaired = ImageEnhance.Contrast(repaired).enhance(1.07)
        repaired = ImageEnhance.Color(repaired).enhance(1.05)

        print("✅ اكتمل الترميم بنجاح!")
        print("═" * 70)

        return repaired.convert("RGB")

    def dna_zoom_repair(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        net: PILImage.Image,
        zoom_factor: float = 1.5,
        strength: float = 0.32,
        steps: int = 28,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.85,
        prompt: str = "",
        negative_prompt: str = "",
        feather_radius: int = 8,          # توسيع خفيف للماسك عند اللصق
        debug: bool = False,
    ) -> PILImage.Image:
        """
        إصلاح المناطق المقنعة بتقنية DNA.Zoom:
        1. استخراج المنطقة + تكبيرها
        2. إعادة توليد التفاصيل باستخدام net كـ control
        3. تصغير وإعادة لصق مع feathering لتجنب الحواف القاسية

        Args:
            zoom_factor: عامل التكبير (1.3–2.0 عادةً)
            strength: قوة الـ denoising في المنطقة المكبرة
            feather_radius: نصف قطر الـ Gaussian blur للـ feathering (تجنب الخطوط الواضحة)
        """
        if debug:
            print("\n" + "="*75)
            print(f"بدء dna_zoom_repair | zoom={zoom_factor:.2f} | strength={strength:.2f}")
            print("="*75)

        # ─── 1. التحضير والتحقق ─────────────────────────────────────────────────
        img   = img.convert("RGB")
        mask  = mask.convert("L")
        net   = net.convert("RGB")

        if not (img.size == mask.size == net.size):
            raise ValueError("أحجام img / mask / net غير متساوية")

        # إيجاد bounding box للمنطقة المقنعة
        bbox = mask.getbbox()
        if bbox is None:
            if debug:
                print("→ الماسك فارغ → إرجاع الصورة كما هي")
            return img

        x1, y1, x2, y2 = bbox
        crop_w = x2 - x1
        crop_h = y2 - y1

        # ─── 2. قص المنطقة + تكبير ──────────────────────────────────────────────
        cropped_img  = img.crop(bbox)
        cropped_mask = mask.crop(bbox)
        cropped_net  = net.crop(bbox)

        new_w = int(crop_w * zoom_factor)
        new_h = int(crop_h * zoom_factor)

        zoomed_img  = cropped_img.resize((new_w, new_h), PILImage.LANCZOS)
        zoomed_mask = cropped_mask.resize((new_w, new_h), PILImage.NEAREST)
        zoomed_net  = cropped_net.resize((new_w, new_h), PILImage.LANCZOS)

        # توسيع الماسك قليلاً لتجنب حواف قاسية بعد اللصق
        if feather_radius > 0:
            zoomed_mask = zoomed_mask.filter(PILImageFilter.GaussianBlur(1.2))
            zoomed_mask = ImageOps.expand(zoomed_mask, border=feather_radius, fill=0)
            zoomed_mask = zoomed_mask.resize((new_w, new_h), PILImage.NEAREST)

        # ─── 3. إعداد الـ prompt إذا لم يُمرر ────────────────────────────────────
        if not prompt:
            prompt = "highly detailed, realistic texture, sharp focus, best quality"

        if not negative_prompt:
            negative_prompt = "blurry, low resolution, artifacts, deformed, bad anatomy"

        # ─── 4. الاستدلال داخل المنطقة المكبرة ──────────────────────────────────
        try:
            pipe_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                # "image": zoomed_img,          ← احذفه مؤقتًا
                # "mask_image": zoomed_mask,    ← احذفه مؤقتًا
                "control_image": zoomed_net,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                # "strength": strength,         ← مش ضروري هنا، لكن لو عايز تضيفه جرب
            }

            with torch.inference_mode():
                output = self.pipeline(**pipe_kwargs)

            generated = output.images[0]

            # لو عايز نحافظ على بعض التفاصيل من الصورة الأصلية المكبرة
            # ممكن ندمجها مع النتيجة بـ PIL بعدين (اختياري)
            # generated = Image.blend(generated, zoomed_img, alpha=0.35)

        except Exception as e:
            print(f"!! خطأ في dna_zoom_repair: {type(e).__name__}: {e}")
            generated = zoomed_img  # fallback بسيط

        # ─── 5. تصغير النتيجة ولصقها مرة أخرى ───────────────────────────────────
        generated = generated.resize((crop_w, crop_h), PILImage.LANCZOS)

        result = img.copy()
        result.paste(generated, (x1, y1), generated)   # paste بدون ماسك إضافي

        # feathering خفيف على النتيجة النهائية (اختياري)
        if feather_radius > 0:
            result = result.filter(PILImageFilter.GaussianBlur(0.6))

        if debug:
            print("→ dna_zoom_repair انتهى بنجاح")
            print("="*75 + "\n")

        return result.convert("RGB")

# ────────────────────────────────────────────────
# استخدام سهل
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # طباعة معلومات PyTorch بشكل آمن
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device index:", torch.cuda.current_device())
        print("CUDA version:", torch.version.cuda)
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
        except RuntimeError as re:
            print("خطأ في الحصول على اسم الكارت:", str(re))
    else:
        print("لا يوجد GPU متاح، سيتم التشغيل على CPU")

    try:
        # فتح الصورة (استخدم Image أو PILImage حسب اللي اخترته فوق)
        img = PILImage.open("input.jpg").convert("RGB")
        print(f"تم فتح الصورة بنجاح: {img.size}")

        repair_system = DNANetPulseRepair()
        print("تم إنشاء DNANetPulseRepair بنجاح")

        result = repair_system.repair(
            img,
            prompt="highly detailed, realistic skin, vibrant colors",
            use_colored_layers=True,
            use_color_pulsing=True,
            pulse_steps=7
        )

        output_path = "repaired_output.jpg"
        result.save(output_path)
        print(f"✅ تم الإصلاح وحفظ النتيجة في: {output_path}")

    except FileNotFoundError:
        print("خطأ: الصورة 'input.jpg' مش موجودة في المجلد الحالي")
        print("ضع صورة باسم input.jpg أو غيّر المسار في الكود")
    except NameError as ne:
        print(f"خطأ في المتغيرات: {ne}")
        print("تأكد من استيراد PIL.Image أو torch في أعلى الملف")
    except Exception as e:
        print(f"خطأ أثناء التشغيل: {type(e).__name__}: {str(e)}")

    # ────────────────────────────────────────────────
    # 1. إنشاء محرك الإصلاح
    # ────────────────────────────────────────────────
    try:
        repair_system = DNANetPulseRepair()
    except Exception as e:
        print("خطأ أثناء إنشاء DNANetPulseRepair:")
        print(type(e).__name__, ":", str(e))
        exit(1)

    # ────────────────────────────────────────────────
    # 2. فتح الصورة
    # ────────────────────────────────────────────────
    input_path = "input.jpg"   # ← يمكنك تغييره هنا أو جعله argument لاحقًا

    try:
        image = PILImage.open(input_path).convert("RGB")
        print(f"تم فتح الصورة: {input_path}  ({image.size})")
    except FileNotFoundError:
        print(f"خطأ: الملف '{input_path}' غير موجود في المجلد الحالي")
        print("ضع صورة باسم input.jpg أو غيّر المسار في الكود")
        exit(1)
    except Exception as e:
        print("خطأ أثناء فتح الصورة:")
        print(type(e).__name__, ":", str(e))
        exit(1)

    # ────────────────────────────────────────────────
    # 3. تنفيذ الإصلاح
    # ────────────────────────────────────────────────
    try:
        result = repair_system.repair(
            image,
            prompt="highly detailed, realistic skin, vibrant colors",
            use_colored_layers=True,
            use_color_pulsing=True,
            pulse_steps=7
        )
        output_path = "dna_net_pulse_repaired.jpg"
        result.save(output_path)
        print(f"✅ تم الترميم بنجاح → {output_path}")
    except AttributeError as ae:
        print("خطأ في الوصول إلى خاصية (غالباً controlnet أو pipeline):")
        print(ae)
    except Exception as e:
        print("خطأ أثناء تنفيذ .repair():")
        print(type(e).__name__, ":", str(e))

    # ────────────────────────────────────────────────
    # 4. اختبار محرك الألوان (آمن من الأخطاء)
    # ────────────────────────────────────────────────
    try:
        color_engine = DndSeedColorEngine()

        print("\n" + "═" * 60)
        print(" " * 15 + "اختبار محرك الألوان – جميع العناصر الـ ٧")
        print("═" * 60)

        seeds = {}
        for element in color_engine.elements:
            seed = color_engine.generate_dnd_seed_color(element)
            seeds[element] = seed
            hex_val = DND_COLOR_MENU[element]['hex']
            print(f"Seed ({element:8}): {seed}  → {hex_val}")

        # ١. متوسط وزني لكل الـ ٧ ألوان (مرة واحدة فقط)
        print("\n" + "═" * 60)
        print("متوسط وزني لكل الـ ٧ ألوان (weighted average)")
        print("═" * 60)

        weights = [1.0] * 7  # كل لون بنفس الوزن
        total = sum(weights)
        r = g = b = 0.0

        for i, el in enumerate(color_engine.elements):
            col = seeds[el]
            w = weights[i] / total
            r += col[0] * w
            g += col[1] * w
            b += col[2] * w

        average_color = (int(round(r)), int(round(g)), int(round(b)))
        print(f"اللون المركزي (متوسط الـ ٧): {average_color}")

        # ٢. مزج عشوائي بين اتنين (مثال واحد فقط)
        print("\n" + "═" * 60)
        print("مزج عشوائي بين اتنين من العناصر (مثال)")
        print("═" * 60)

        el1, el2 = random.sample(color_engine.elements, 2)
        col1 = seeds[el1]
        col2 = seeds[el2]

        mixed = color_engine.mix_dnd_seed_colors(col1, col2, ratio=0.45)
        print(f"Mixed {el1:8} + {el2:8} (45%): {mixed}")

        if hasattr(color_engine, "monitor_dnd_color_mix"):
            report = color_engine.monitor_dnd_color_mix(col1, col2, mixed, ratio=0.45)
            print("\nتقرير المزج:")
            for k, v in report.items():
                print(f"  {k}: {v}")
        else:
            print("دالة monitor_dnd_color_mix غير موجودة حاليًا")

    except Exception as e:
        print("خطأ عام في محرك الألوان:")
        print(type(e).__name__, str(e))

        # __________________________________  اختبار سريع __________________________________

        print("\n=== اختبار سريع ===")
    try:
        from PIL import Image
        img = PILImage.open("input.jpg").convert("RGB")
        print(f"تم فتح الصورة: {img.size}")

        # لو الدالة repair موجودة ومكتملة
        if hasattr(repair_system, "repair"):
            result = repair_system.repair(
                img,
                prompt="highly detailed, realistic skin, vibrant colors",
                use_colored_layers=True,
                use_color_pulsing=True,
                pulse_steps=7
            )
            result.save("test_repaired.jpg")
            print("✅ تم حفظ النتيجة: test_repaired.jpg")
        else:
            print("دالة .repair() غير موجودة أو غير مكتملة بعد")

    except FileNotFoundError:
        print("ضع صورة باسم input.jpg في نفس المجلد")
    except Exception as e:
        print("خطأ في الاختبار:")
        print(type(e).__name__, str(e))
