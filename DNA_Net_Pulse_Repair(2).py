# DNA_Net_Pulse_Repair.py
"""
DNA_Repair_Pipeline.py

النهج الجديد: DNA-inspired Layer + ControlNet-guided Net
مع إسقاط نبضي DNA-inspired coloring (موجب/سالب) + طبقات لونية مخصصة
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple, List, Dict
import numpy as np
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageFilter, ImageDraw
import cv2
import random
import colorsys
import os
from scipy.ndimage import gaussian_filter

# ─── Diffusers & ControlNet ────────────────────────────────────────
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel

# ─── جهاز و dtype مركزيين ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

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

    def generate_dnd_seed_color(self, element="random", variation=0.12, brightness_boost=0.0):
        if element == "random" or element not in self.color_menu:
            element = random.choice(self.elements)

        base = self.color_menu[element]["base_rgb"]   # ← تأكد أن هذا tuple (r,g,b)
        # ... باقي الحسابات

        r = int(base[0] * (1 + random.uniform(-variation, variation)))
        g = int(base[1] * (1 + random.uniform(-variation, variation)))
        b = int(base[2] * (1 + random.uniform(-variation, variation)))

        r = int(r * (1 + brightness_boost))
        g = int(g * (1 + brightness_boost))
        b = int(b * (1 + brightness_boost))

        color = tuple(np.clip([r, g, b], 0, 255).astype(int))

        print(f"generate_dnd_seed_color → {element:8} → {color}")   # debug
        return color

    # ──────────────────────────────────────────────────────────────
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
        """التأكد من أن اللون tuple من 3 أعداد صحيحة في النطاق 0–255"""
        if not isinstance(color, tuple) or len(color) != 3:
            raise TypeError(f"{param_name} يجب أن يكون tuple مكون من 3 أعداد")

        # تحويل أي np.int64 أو np.integer إلى int عادي + unpacking لضمان الطول
        try:
            r, g, b = (int(v) for v in color)
        except (ValueError, TypeError):
            raise ValueError(f"لا يمكن تحويل قيم {param_name} إلى أعداد صحيحة: {color}")

        if not all(isinstance(v, (int, float, np.integer)) and 0 <= v <= 255 for v in col):
            raise ValueError(f"قيم {name} غير صالحة: {col}")

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

    # ──────────────────────────────────────────────────────────────
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

    def add_dna_colored_layers(
        self,
        net_image: Image.Image,
        mask: Image.Image,
        colors: List[Tuple[int, int, int]] = [(255, 80, 80), (80, 255, 80), (80, 80, 255)],
        blend_mode: Literal["density", "wave"] = "density",
        opacity: float = 0.50,   # ← لاحظ: هنا opacity مش base_opacity
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
        # التأكد من الصيغة (بدون الاعتماد على دالة _ensure_rgb غير موجودة)
        img = img.convert("RGB") if img.mode != "RGB" else img
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

        # التأكد من توافق الأحجام قبل الـ pipeline
        if img.size != mask.size or img.size != control_img.size:
            raise ValueError(
                f"أحجام غير متطابقة: img={img.size}, mask={mask.size}, control_img={control_img.size}"
            )

        with torch.no_grad(), torch.inference_mode():
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
                # variant=variant,               ← احذف أو علّق السطر ده
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
                # variant=variant,               ← احذف أو علّق هنا كمان
            )
            print("✓ تم التحميل من Hugging Face")

        self.controlnet = self.controlnet.to(self.device)
        print(f"ControlNet موجود على: {next(self.controlnet.parameters()).device}")

        # ────────────────────────────────────────────────
        # Section 5: Pipeline Loader
        # ────────────────────────────────────────────────
        print("\n=== تحميل Pipeline ===")

        # أول حاجة: نتأكد إن controlnet اتحمل فعلاً
        if not hasattr(self, 'controlnet'):
            raise RuntimeError("فشل تحميل ControlNet تمامًا قبل محاولة إنشاء Pipeline")

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

            try:
                self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    controlnet=self.controlnet,
                    torch_dtype=self.dtype,
                    variant=variant,
                    safety_checker=None,
                )
                print("✓ Pipeline (fallback) تم تحميله")

            except Exception as fallback_err:
                raise RuntimeError(
                    f"فشل تحميل الـ Pipeline كليًا (حتى الـ fallback):\n"
                    f"الخطأ الأساسي: {e}\n"
                    f"خطأ الـ fallback: {fallback_err}"
                )

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

    # ────────────────────────────────────────────────
    # كشف المناطق "الميتة" أو المنهارة / منخفضة التفاصيل
    # ────────────────────────────────────────────────
    def detect_dead_zones(
            self,
            img: Image.Image,
            method: str = "multi",                  # "multi", "canny_dilate", "laplacian_var", "entropy"
            control_type: str = "tile",             # للتوافق مع استدعاءات أخرى (غير مستخدم حاليًا)
            canny_low: int = 60,
            canny_high: int = 180,
            dilation_kernel_size: int = 9,
            threshold: float = 0.25,
            min_area_ratio: float = 0.008,
            return_type: str = "mask"               # "mask", "score_map", "signed_map"
        ) -> Image.Image:
        """
        كشف المناطق الميتة بمزيج من المقاييس + تنظيف ماسك نهائي
        يرجع ماسك L (0=سليم، 255=ميت) أو خريطة درجات أو تصور ملون
        """
        # ────────────────── الجزء 1: التحضير الأساسي ──────────────────
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # ────────────────── الجزء 2: حساب المقاييس المشتركة ──────────────────
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 2.1 تباين لابلاسيان (جيد للكشف عن الـ blur / low-detail)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_abs = np.abs(lap)
        lap_var = gaussian_filter(lap_abs, sigma=1.0)          # أكثر استقرارًا من GaussianBlur في بعض الحالات
        contrast_score = lap_var / (lap_var.max() + 1e-8)

        # 2.2 كثافة الحواف (Canny)
        edges = cv2.Canny(gray.astype(np.uint8), canny_low, canny_high)
        edge_density = cv2.GaussianBlur(
            edges.astype(np.float32) / 255.0,
            (7, 7),
            sigmaX=0
        )

        # 2.3 التشبع والسطوع (HSV) – نستخدم float32 من البداية
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_score = hsv[..., 1] / 255.0
        val_score = hsv[..., 2] / 255.0

        # ────────────────── الجزء 3: اختيار طريقة الحساب ──────────────────
        if method == "multi":
            # مزيج متوازن (يمكن تعديل الأوزان بسهولة هنا مستقبلاً)
            final_score = (
                0.40 * contrast_score +     # أهمية عالية للتباين
                0.30 * edge_density +       # الحواف مهمة للهيكل
                0.20 * sat_score +          # تشبع منخفض = ميت غالباً
                0.10 * val_score            # سطوع متطرف = مشكلة
            )

        elif method == "canny_dilate":
            # الطريقة القديمة المفضلة
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            edge_density = dilated.astype(np.float32) / 255.0
            final_score = 1.0 - edge_density

        elif method == "laplacian_var":
            final_score = contrast_score  # أو 1.0 - contrast_score لو عايز عكس المنطق

        elif method == "entropy":
            # placeholder – يمكن تنفيذه لاحقاً بـ skimage أو نافذة منزلقة
            final_score = contrast_score  # مؤقتاً

        else:
            raise ValueError(f"طريقة غير مدعومة: {method}")

        # ─────────────── الجزء 4: تطبيع إلى نطاق سالب/موجب حقيقي ───────────────
        final_score = (final_score - final_score.mean()) / (final_score.std() + 1e-8)
        final_score = np.clip(final_score, -1.5, 1.5)

        # ──────────── الجزء 5: تنظيف الماسك (مهم جدًا – من القديمة) ───────────────
        dead_mask = (final_score < -threshold).astype(np.uint8) * 255

        # إزالة المناطق الصغيرة جدًا
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dead_mask = cv2.morphologyEx(dead_mask, cv2.MORPH_OPEN, kernel_open)

        # توسيع خفيف للاندماج مع الإصلاح
        dead_mask = cv2.dilate(dead_mask, np.ones((5,5), np.uint8), iterations=1)

        # ──────────────────────── الجزء 6: الإخراج حسب الطلب ────────────────────────
        if return_type == "mask":
            return Image.fromarray(dead_mask).convert("L")

        elif return_type == "score_map":
            vis = ((final_score + 1.5) / 3.0 * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(vis).convert("L")

        elif return_type == "signed_map":
            h = np.zeros_like(gray, dtype=np.uint8)
            s = np.full_like(gray, 255, dtype=np.uint8)
            v = np.full_like(gray, 255, dtype=np.uint8)
            h[final_score < 0] = 0      # أحمر = ميت
            h[final_score >= 0] = 85    # أخضر = حي
            hsv_out = np.stack([h, s, v], axis=-1)
            rgb_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
            return Image.fromarray(rgb_out)

        else:
            raise ValueError("return_type غير مدعوم")

    def _ensure_rgb(self, img: Image.Image) -> Image.Image:
        """
        التأكد من أن الصورة في وضع RGB
        """
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

    # ────────────────────────────────────────────────
    #  توليد طبقة الشبكة الهيكلية (الدالة الرئيسية)
    # ────────────────────────────────────────────────
    def generate_net_structure(
        self,
        img: Image.Image,
        mask: Image.Image,
        control_type: str = "canny",
        net_strength: float = 0.65,
        steps: int = 20,
        canny_low: int = 70,
        canny_high: int = 170,
        lineart_thickness: int = 2,
    ) -> Image.Image:
        """
        إنشاء طبقة شبكة هيكلية (net) باستخدام ControlNet
        تدعم حالياً: canny, lineart, union
        """
        img = self._ensure_rgb(img)
        mask = mask.convert("L")

        # 1. تحضير صورة التحكم حسب النوع المطلوب
        control_image = self._prepare_control_image(
            img=img,
            control_type=control_type,
            canny_low=canny_low,
            canny_high=canny_high,
            lineart_thickness=lineart_thickness,
        )

        # 2. إعداد الـ prompt والـ negative prompt (يمكن فصلهما لاحقاً إلى متغيرات كلاس)
        positive_prompt = self._get_net_positive_prompt()
        negative_prompt = self._get_net_negative_prompt()

        # 3. تنفيذ الـ inference
        generated = self._run_controlnet_inference(
            image=img,
            mask_image=mask,
            control_image=control_image,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            strength=net_strength,
            steps=steps,
            guidance_scale=7.0,
        )

        return generated


    def _prepare_control_image(
        self,
        img: Image.Image,
        control_type: str,
        canny_low: int,
        canny_high: int,
        lineart_thickness: int,
    ) -> Image.Image:
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

    def _create_canny_control(self, img_np: np.ndarray, low: int, high: int) -> Image.Image:
        """إنشاء control image بنمط Canny مع dilation خفيف"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)

        # توسيع خفيف لتقليل الثغرات
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return Image.fromarray(edges).convert("RGB")

    def _create_lineart_control(self, img_np: np.ndarray, thickness: int) -> Image.Image:
        """إنشاء control image بنمط lineart باستخدام adaptive threshold"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        block_size = 11 + thickness * 2
        lineart = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=2
        )

        # عكس الألوان (الخطوط بيضاء على خلفية سوداء → العكس)
        lineart = 255 - lineart

        return Image.fromarray(lineart).convert("RGB")

    def _get_net_positive_prompt(self) -> str:
        """النص الإيجابي المستخدم لتوليد الشبكة الهيكلية"""
        return (
            "structural grid, clean architectural lines, "
            "technical blueprint style, high contrast edges, "
            "precise geometric network, technical drawing, "
            "sharp vector lines, schematic, diagram"
        )

    def _get_net_negative_prompt(self) -> str:
        """النص السلبي لتجنب العيوب الشائعة"""
        return (
            "blurry, noisy, low detail, artifacts, text, watermark, "
            "overexposed, underexposed, deformed, low quality, "
            "bad anatomy, jpeg artifacts, compression, grainy"
        )

    def _run_controlnet_inference(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        """تنفيذ الـ inference باستخدام الـ pipeline مع حماية الذاكرة"""
        with torch.no_grad(), torch.inference_mode():
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_image,
                control_image=control_image,
                controlnet_conditioning_scale=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            ).images[0]

        return output

    # ====================== 2. DNA Layer شفافة أولى ======================
    def create_dna_base_layer(
            self,
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

    # ====================== DNA Pulse Repair Layer (نبضات موجب/سالب) ======================
    def create_dna_pulse_repair_layer(
            self,
            size: Tuple[int, int],
            positive_pulse: Tuple[int, int, int] = (20, 15, 10),
            negative_pulse: Tuple[int, int, int] = (-15, -10, -5),
            pulse_steps: int = 5,
            initial_opacity: float = 0.40,
            opacity_decay: float = 0.06,
        ) -> Image.Image:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")

        for step in range(pulse_steps):
            alpha = int(255 * (initial_opacity - step * opacity_decay))
            if alpha <= 0:
                break

            # نبضة موجبة
            pos_rgb: Tuple[int, int, int] = tuple(int(c * (1 + step * 0.08)) for c in positive_pulse)
            pos_fill: Tuple[int, int, int, int] = pos_rgb + (alpha,)
            draw.rectangle((0, 0, size[0], size[1]), fill=pos_fill)

            # نبضة سالبة
            neg_rgb: Tuple[int, int, int] = tuple(int(c * (1 - step * 0.1)) for c in negative_pulse)
            neg_fill: Tuple[int, int, int, int] = neg_rgb + (alpha // 2,)
            draw.rectangle((0, 0, size[0], size[1]), fill=neg_fill)

            # بعد detect_dead_zones
            dead_mask = self.detect_dead_zones(
                img,
                method="multi",
                threshold=0.25,
                return_type="mask"
            )

            # أو لو عايز خريطة سالب/موجب ملونة للتصور (اختياري)
            signed_map = self.detect_dead_zones(
                img,
                method="multi",
                return_type="signed_map"
            )
            signed_map.save("debug_signed_map.jpg")  # للمعاينة

            # ─── معالجة المناطق السالبة بالنبضة الموجبة ───
            print("  قبل create_dna_pulse_repair_layer")
            pulse_layer = self.create_dna_pulse_repair_layer(
                size=img.size,
                positive_pulse=(30, 25, 20),     # قيم موجبة أقوى
                negative_pulse=(-10, -8, -5),    # سالبة خفيفة للتوازن
                pulse_steps=6,
                initial_opacity=0.45,
                opacity_decay=0.05,
            )

            # دمج النبضة على المناطق السالبة فقط (باستخدام الماسك)
            print("  قبل دمج النبضة")
            repaired = Image.composite(pulse_layer, img.convert("RGBA"), dead_mask.convert("L"))
            print("  بعد دمج النبضة → repaired تم تحديثه")

            # أو لو عايز alpha_composite بدل composite (لو النبضة شفافة)
            # repaired = Image.alpha_composite(img.convert("RGBA"), pulse_layer)

        return layer

    def create_dna_enhancement_stack(
            self,
            size: Tuple[int, int]
        ) -> Image.Image:
        """
        دمج الطبقة الأساسية DNA مع طبقة النبض الإصلاحي
        لإنشاء طبقة تعزيز كاملة (stack) DNA-inspired
        """
        base = self.create_dna_base_layer(size, opacity=0.35)
        pulse = self.create_dna_pulse_repair_layer(size, pulse_steps=5)

        return Image.alpha_composite(base, pulse)

    # ====================== 3. إعادة ترميم الشكل الهندسي ======================
    def repair_geometry_with_net(
            self,
            img: Image.Image,
            mask: Image.Image,
            net: Image.Image,
            prompt: str = "high quality, detailed geometry, realistic structure, precise lines, architectural accuracy",
            strength: float = 0.32,
            steps: int = 22,
        ) -> Image.Image:
        """
        إعادة بناء الشكل الهندسي بدقة عالية باستخدام الشبكة اللي تم توليدها
        """
        img = self._ensure_rgb(img)
        mask = mask.convert("L")
        net = self._ensure_rgb(net)

        # ─── الترميم داخل no_grad + inference_mode ───
        with torch.no_grad(), torch.inference_mode():  # type: ignore[attr-defined]
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=(
                    "blurry, deformed, low detail, artifacts, noise, text, watermark, "
                    "distorted geometry, bad anatomy, low quality, overexposed, underexposed"
                ),
                image=img,
                mask_image=mask,
                control_image=net,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=7.5,
            )
            result = output.images[0]  # type: ignore[attr-defined]

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
        # التأكد من أن الصورة RGBA
        net_image = net_image.convert("RGBA")
        net_arr = np.array(net_image, dtype=np.float32)

        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

        # طبقة جديدة شفافة
        layer = np.zeros_like(net_arr, dtype=np.uint8)

        # استخراج أضلاع الشبكة
        gray = cv2.cvtColor(net_arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 160)

        if blend_mode == "density":
            intensity = mask_arr ** 1.4
            for i, color in enumerate(base_colors):
                weight = intensity * (0.7 - i * 0.18)
                layer[..., :3] += (np.array(color, dtype=np.float32) * weight[..., None]).astype(np.uint8)

        elif blend_mode == "wave":
            h, w = mask_arr.shape
            wave = np.sin(np.linspace(0, 25 * np.pi, w) + np.arange(h)[:, None] * 0.08)
            wave = (wave + 1) / 2
            wave = np.repeat(wave[..., None], 3, axis=2)
            mixed = np.zeros((h, w, 3), dtype=np.float32)
            for i, color in enumerate(base_colors):
                mixed += np.array(color, dtype=np.float32) * wave * (1 / len(base_colors))
            layer[..., :3] = mixed.astype(np.uint8)

        # طبقة الشفافية
        layer[..., 3] = (mask_arr * 255 * opacity).clip(0, 255).astype(np.uint8)

        # تحسين خطوط الـ Net
        layer[edges > 0, :3] = (220, 220, 255)
        layer[edges > 0, 3] = 240

        # التأكد من عدم تجاوز النطاق
        layer = np.clip(layer, 0, 255).astype(np.uint8)

        return Image.fromarray(layer)

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
        img: Image.Image,
        mask: Image.Image,
        pulse_steps: int = 6,
        hue_std_base: float = 6.0,
        positive_sat_boost: float = 0.25,
        negative_sat_suppress: float = 0.18,
        positive_val_boost: float = 0.15,
        factor_decay: float = 0.65,
        clip_hue: bool = True,
    ) -> PILImage.Image:
        """
        نبض DNA-inspired كامل – نسخة محسّنة مع طباعة تشخيصية مفصلة
        """
        print("\n" + "="*60)
        print("بدء dna_full_pulse")
        print(f"  pulse_steps     = {pulse_steps}")
        print(f"  hue_std_base    = {hue_std_base}")
        print(f"  sat_boost       = {positive_sat_boost:.3f}")
        print(f"  sat_suppress    = {negative_sat_suppress:.3f}")
        print(f"  val_boost       = {positive_val_boost:.3f}")
        print(f"  factor_decay    = {factor_decay:.3f}")
        print(f"  clip_hue        = {clip_hue}")
        print("="*60)

        # ─── التحضير الأولي ─────────────────────────────────────────────
        rgb = np.array(img.convert("RGB"), dtype=np.float32)
        print(f"  حجم الصورة RGB     : {rgb.shape}  dtype={rgb.dtype}")

        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        mask_arr = np.expand_dims(mask_arr, axis=-1)
        print(f"  حجم الماسك         : {mask_arr.shape}  mean={mask_arr.mean():.4f}")

        # ─── تحويل HSV مرة واحدة ────────────────────────────────────────
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        print(f"  تحويل إلى HSV      : {hsv.shape}")
        print(f"     Hue   mean/min/max : {hsv[...,0].mean():6.2f}  {hsv[...,0].min():6.2f} → {hsv[...,0].max():6.2f}")
        print(f"     Sat   mean/min/max : {hsv[...,1].mean():6.2f}  {hsv[...,1].min():6.2f} → {hsv[...,1].max():6.2f}")
        print(f"     Val   mean/min/max : {hsv[...,2].mean():6.2f}  {hsv[...,2].min():6.2f} → {hsv[...,2].max():6.2f}")

        # ─── الحلقة الرئيسية ────────────────────────────────────────────
        for step in range(pulse_steps):
            factor = max(0.0, 1.0 - (step / pulse_steps) * factor_decay)
            print(f"\n  ┌─ Step {step+1}/{pulse_steps}   factor = {factor:.4f}")

            # Hue
            hue_shift = np.random.normal(0, hue_std_base * factor, size=(hsv.shape[0], hsv.shape[1]))
            hue_shift *= mask_arr[..., 0]

            print(f"  │   Hue shift   mean/std/min/max : "
                f"{hue_shift.mean():+6.3f} / {hue_shift.std():5.3f}   "
                f"{hue_shift.min():+6.2f} → {hue_shift.max():+6.2f}")

            hsv[..., 0] += hue_shift

            if clip_hue:
                hsv[..., 0] = np.mod(hsv[..., 0], 180.0)
                method = "np.mod"
            else:
                hsv[..., 0] = hsv[..., 0] % 180.0
                method = "% 180"

            print(f"  │   Hue بعد {method:>6}  mean/min/max : "
                f"{hsv[...,0].mean():6.2f}  {hsv[...,0].min():6.2f} → {hsv[...,0].max():6.2f}")

            # Saturation
            saturation_original = hsv[..., 1] / 255.0
            neutral_mask = (saturation_original < 0.12).astype(np.float32)

            sat_mult = 1.0 + (positive_sat_boost - negative_sat_suppress) * factor * mask_arr[..., 0]
            sat_mult *= (1.0 - neutral_mask * 0.7)
            sat_mult = np.clip(sat_mult, 0.15, 2.8)

            print(f"  │   sat_mult    mean/min/max : "
                f"{sat_mult.mean():.4f}  {sat_mult.min():.4f} → {sat_mult.max():.4f}")

            hsv[..., 1] *= sat_mult
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

            print(f"  │   Saturation بعد التعديل  mean/min/max : "
                f"{hsv[...,1].mean():6.2f}  {hsv[...,1].min():6.2f} → {hsv[...,1].max():6.2f}")

            # Value
            val_mult = 1.0 + positive_val_boost * factor * mask_arr[..., 0]
            val_mult *= (1.0 - neutral_mask * 0.6)
            val_mult = np.clip(val_mult, 0.70, 1.45)

            print(f"  │   val_mult    mean/min/max : "
                f"{val_mult.mean():.4f}  {val_mult.min():.4f} → {val_mult.max():.4f}")

            hsv[..., 2] *= val_mult
            hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

            print(f"  │   Value بعد التعديل      mean/min/max : "
                f"{hsv[...,2].mean():6.2f}  {hsv[...,2].min():6.2f} → {hsv[...,2].max():6.2f}")
            print("  └───────────────────────────────────────")

        # ─── الإخراج النهائي ────────────────────────────────────────────
        print("\n  التحويل النهائي HSV → RGB")
        rgb_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)

        print(f"  rgb_out shape/dtype : {rgb_out.shape}  {rgb_out.dtype}")
        print(f"  R mean/min/max      : {rgb_out[...,0].mean():6.2f}  {rgb_out[...,0].min()} → {rgb_out[...,0].max()}")
        print(f"  G mean/min/max      : {rgb_out[...,1].mean():6.2f}  {rgb_out[...,1].min()} → {rgb_out[...,1].max()}")
        print(f"  B mean/min/max      : {rgb_out[...,2].mean():6.2f}  {rgb_out[...,2].min()} → {rgb_out[...,2].max()}")

        result = Image.fromarray(rgb_out)
        print("  تم إرجاع الصورة الناتجة")
        print("="*60 + "\n")

        return result

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
        print("دخول repair() – بداية الدالة")

        # 1. اكتشاف المناطق الميتة / الماسك
        print("  قبل detect_dead_zones (الأولى)")
        mask = self.detect_dead_zones(img, control_type="tile", threshold=0.35)
        print(f"  بعد detect_dead_zones → mask: {mask is not None}, size={mask.size if mask else 'None'}")

        # 2. توليد الشبكة الهيكلية (Net)
        print("  قبل generate_net_structure")
        net = self.generate_net_structure(
            img=img,
            mask=mask,
            control_type="union",
            net_strength=0.68,
            steps=18,
        )
        print(f"  بعد generate_net_structure → net: {net is not None}, size={net.size if net else 'None'}")

        # احتمال وجود assert هنا في الكود الأصلي
        print("  بعد توليد net – هل في assert على net؟ (لو ما ظهرش اللي بعده يبقى فيه assert)")

        # 3. طبقة DNA أساسية شفافة
        print("  قبل create_dna_base_layer")
        dna_base = self.create_dna_base_layer(img.size, opacity=0.38)
        print(f"  بعد create_dna_base_layer → dna_base: {dna_base is not None}, size={dna_base.size if dna_base else 'None'}")

        # احتمال assert هنا (لو الكود بيتحقق من الطبقة)
        print("  بعد dna_base – هل في assert على dna_base؟")

        # 4. الترميم الهندسي الرئيسي
        print("  قبل repair_geometry_with_net")
        repaired = self.repair_geometry_with_net(
            img=img,
            mask=mask,
            net=net,
            prompt=prompt,
            strength=0.35,
            steps=25,
        )
        print(f"  بعد repair_geometry_with_net → repaired: {repaired is not None}, size={repaired.size if repaired else 'None'}")

        # احتمال assert هنا (غالباً ده المكان المشتبه فيه)
        print("  بعد repaired – هل في assert على repaired؟ (لو ما ظهرش اللي بعده يبقى فيه assert هنا)")

        # 5. طبقات لونية (اختياري)
        if use_colored_layers:
            print("  قبل add_dna_colored_layers")
            colored = self.add_dna_colored_layers(
                net_image=net,
                mask=mask,
                base_colors=[(220, 60, 30), (40, 180, 60), (80, 180, 255)],   # ← غيّر colors → base_colors
                blend_mode="density",
                opacity=0.55,
            )
            print(f"  بعد add_dna_colored_layers → colored: {colored is not None}, size={colored.size if colored else 'None'}")

            print("  قبل alpha_composite")
            repaired = Image.alpha_composite(repaired.convert("RGBA"), colored)
            print("  بعد alpha_composite → repaired تم تعديله")
        else:
            print("  use_colored_layers = False → تم تخطي الطبقات اللونية")

        # 6. نبض لوني DNA-inspired (اختياري)
        print("  قبل dna_full_pulse (إن وُجد)")
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
            print(f"  بعد dna_full_pulse → final: {final is not None}, size={final.size if final else 'None'}")
        else:
            final = repaired
            print("  use_color_pulsing = False → final = repaired")

        # 7. ماسك ثانٍ + تلميع نهائي
        print("  قبل detect_dead_zones (الثانية)")
        mask = self.detect_dead_zones(
            img,
            method="canny_dilate",
            canny_low=50,
            canny_high=160,
            dilation_kernel_size=11,
            threshold=0.25
        )
        print(f"  بعد detect_dead_zones الثانية → mask: {mask is not None}, size={mask.size if mask else 'None'}")

        print("  قبل التلميع النهائي")
        final = ImageEnhance.Sharpness(final).enhance(1.10)
        final = ImageEnhance.Contrast(final).enhance(1.05)
        print("  بعد التلميع النهائي")

        print("  خروج repair() – نجاح (لو وصلنا هنا يبقى مفيش assert وقّفنا)")
        return final.convert("RGB")

# ────────────────────────────────────────────────
# استخدام سهل
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from PIL import Image

    # طباعة معلومات PyTorch بشكل آمن (تجنب أخطاء Pylance أو import غير كامل)
    print("PyTorch version:", getattr(torch, "__version__"))
    print("CUDA available:", getattr(torch, "cuda").is_available())

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

    # باقي الكود (إنشاء repair_system، فتح الصورة، الإصلاح، إلخ...)
    try:
        repair_system = DNANetPulseRepair()
    except Exception as e:
        print("خطأ أثناء إنشاء DNANetPulseRepair:", type(e).__name__, str(e))
        exit(1)

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
        image = Image.open(input_path).convert("RGB")
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

        seed1 = color_engine.generate_dnd_seed_color("Fire")
        seed2 = color_engine.generate_dnd_seed_color("Ice")

        print(f"Seed 1 (Fire): {seed1}  → {DND_COLOR_MENU['Fire']['hex']}")
        print(f"Seed 2 (Ice):  {seed2}  → {DND_COLOR_MENU['Ice']['hex']}")

        mixed = color_engine.mix_dnd_seed_colors(seed1, seed2, ratio=0.45)
        print(f"Mixed color: {mixed}")

        # تجنب الخطأ لو الدالة فيها مشكلة حسابية
        if hasattr(color_engine, "monitor_dnd_color_mix"):
            try:
                report = color_engine.monitor_dnd_color_mix(seed1, seed2, mixed, ratio=0.45)
                print("\nتقرير المزج:")
                for k, v in report.items():
                    print(f"  {k}: {v}")
            except TypeError as te:
                print("خطأ في حساب التقرير (ربما قيمة نصية في عملية رياضية):")
                print(te)
        else:
            print("دالة monitor_dnd_color_mix غير موجودة حاليًا")

    except Exception as e:
        print("خطأ عام في محرك الألوان:")
        print(type(e).__name__, str(e))

        # __________________________________  اختبار سريع __________________________________

        print("\n=== اختبار سريع ===")
    try:
        from PIL import Image
        img = Image.open("input.jpg").convert("RGB")
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
