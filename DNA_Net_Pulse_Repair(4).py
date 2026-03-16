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
import PIL
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageFilter, ImageDraw
import cv2
import sys
import random
import os
from scipy.ndimage import gaussian_filter

from union_multi_inpainting import union_img2img_with_mask

# ─── Diffusers & ControlNet ────────────────────────────────────────
from diffusers.pipelines.controlnet.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel

import torch
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
        dtype: Optional[TorchDType] = None,
        color_engine=None,
        enable_attention_slicing: bool = True,
        enable_cpu_offload: bool = True,
        variant: str = "fp16",
    ):

        global torch   # ← هنا بالضبط، أول سطر داخل الدالة

        self.use_dna_zoom = False               # default off
        self.dna_zoom_factor = 1.5
        self.dna_zoom_strength = 0.32
        self.dna_zoom_control_scale = 0.85

        """
        تهيئة محرك الإصلاح DNA-Net-Pulse مع ControlNet Union + SDXL Inpainting

        Args:
            controlnet_model: اسم أو مسار نموذج ControlNet
            sd_model: اسم أو مسار نموذج Stable Diffusion XL
            device: "cuda" أو "cpu" (تلقائي إذا لم يُحدد)
            dtype: torch.float16 أو torch.float32 (تلقائي حسب الجهاز)
            color_engine: كائن DndSeedColorEngine (اختياري)
            enable_attention_slicing: تفعيل تقطيع الـ attention لتوفير VRAM
            enable_cpu_offload: تفعيل نقل النموذج بين CPU/GPU
            variant: "fp16" أو غيره
        """

        # ──────────────────────────────────────────────────────────────
        # 1. تعريف الجهاز والنوع (dtype) أولاً – مهم لـ Pylance
        # ──────────────────────────────────────────────────────────────
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dtype = (
            dtype
            if dtype is not None
            else (torch.float16 if self.device.type == "cuda" else torch.float32)
        )

        print(f"→ Using device: {self.device} | dtype: {self.dtype}")

        print(f"→ Device: {self.device} | dtype: {self.dtype}")

        # ربط محرك الألوان (اختياري)
        self.color_engine = color_engine
        if self.color_engine:
            print("→ Color engine تم ربطه بنجاح")
        else:
            print("→ لا يوجد color engine خارجي → سيتم الاعتماد على الألوان الافتراضية")

        # ──────────────────────────────────────────────────────────────
        # 2. تحميل ControlNet (محلي أولاً ثم من HF)
        # ──────────────────────────────────────────────────────────────
        print("\n=== تحميل ControlNet Union ===")
        self.controlnet = None
        self.pipeline = None
        self.fallback_mode = False  # flag عشان نعرف إذا دخلنا fallback ولا لأ

        # 1. محاولة التحميل من المسار المحلي (أولوية أولى)
        try:
            self.controlnet = ControlNetUnionModel.from_pretrained(
                LOCAL_CONTROLNET_PATH,
                torch_dtype=self.dtype,
                use_safetensors=True,
                local_files_only=True,   # يمنع أي محاولة تحميل من النت
            )
            print("✓ ControlNet Union تم تحميله من المسار المحلي بنجاح")
        except Exception as local_err:
            print(f"⚠ فشل التحميل المحلي: {type(local_err).__name__}: {local_err}")
            print("→ جاري المحاولة من Hugging Face...")

            # 2. fallback من Hugging Face
            try:
                self.controlnet = ControlNetUnionModel.from_pretrained(
                    "xinsir/controlnet-union-sdxl-1.0",
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    use_safetensors=True,
                )
                print("✓ ControlNet Union تم تحميله من Hugging Face بنجاح")
            except Exception as hf_err:
                raise RuntimeError(
                    f"فشل تحميل ControlNet Union كليًا (محلي و HF):\n"
                    f"  Local error: {local_err}\n"
                    f"  HF error: {hf_err}"
                ) from hf_err  # <-- إضافة from لتتبع أفضل في الـ traceback

        # ─── نقل ControlNet للجهاز بعد التحميل الناجح ───────────────────────
        if self.controlnet is not None:
            try:
                self.controlnet = self.controlnet.to(self.device)
                # التحقق الفعلي من الجهاز بعد النقل
                actual_device = next(self.controlnet.parameters()).device
                print(f"ControlNet موجود على: {actual_device}")
                if actual_device.type != self.device.type:
                    print(f"  تحذير: الجهاز الفعلي ({actual_device}) مختلف عن المطلوب ({self.device})")
            except Exception as move_err:
                raise RuntimeError(
                    f"فشل نقل ControlNet إلى الجهاز {self.device}: {move_err}"
                ) from move_err
        else:
            raise RuntimeError("ControlNet لم يتم تحميله نهائيًا – لا يمكن الاستمرار")

        # ──────────────────────────────────────────────────────────────
        # 3. تحميل الـ Pipeline (مرة واحدة فقط)
        # ──────────────────────────────────────────────────────────────
        print("\n=== تحميل StableDiffusionXLControlNetUnionPipeline ===")
        self.pipeline = None

        # المحاولة الأساسية
        try:
            print(f"  محاولة تحميل {sd_model} مع Union Pipeline ...")
            self.pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
                sd_model,
                controlnet=self.controlnet,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
                use_safetensors=True,
                safety_checker=None,
            )
            print("  ✓ تم التحميل الأساسي بنجاح (Union Pipeline)")
        except Exception as pipe_err:
            print(f"  !! فشل التحميل الأساسي: {type(pipe_err).__name__}: {pipe_err}")

            # fallback واحد ومنطقي: نموذج SDXL الأساسي
            try:
                print("  → جاري تجربة fallback بنموذج SDXL الأساسي...")
                self.pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    controlnet=self.controlnet,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    use_safetensors=True,
                    safety_checker=None,
                )
                print("  ✓ تم التحميل بنجاح في الـ fallback (Union Pipeline)")
            except Exception as fb_err:
                raise RuntimeError(
                    f"فشل تحميل Pipeline كليًا:\n"
                    f"  Main error: {pipe_err}\n"
                    f"  Fallback error: {fb_err}"
                )

        # ─── بعد النجاح: نقل + تفعيل التحسينات بترتيب آمن ─────────────────
        if self.pipeline is not None:
            print("→ نقل النموذج وتفعيل التحسينات...")

            # 1. نقل للجهاز أولاً (مهم قبل أي offload)
            self.pipeline.to(self.device)
            print(f"  → النموذج تم نقله إلى: {self.device}")

            # 2. تفعيل attention slicing (خفيف ومفيد دائمًا)
            if enable_attention_slicing:
                print("→ تفعيل attention slicing (لتوفير VRAM)")
                try:
                    self.pipeline.enable_attention_slicing("max")
                    print("  ✓ تم تفعيل attention slicing بنجاح")
                except Exception as slice_err:
                    print(f"  !! فشل تفعيل attention slicing: {slice_err}")

            # 3. تفعيل CPU offload (لو cuda)
            if enable_cpu_offload and self.device.type == "cuda":
                print("→ تفعيل model CPU offload (لتوفير VRAM)")
                try:
                    self.pipeline.enable_model_cpu_offload()
                    print("  ✓ تم تفعيل CPU offload بنجاح")
                except Exception as offload_err:
                    print(f"  !! فشل تفعيل CPU offload: {offload_err}")
                    print("  → سيتم الاستمرار بدون offload (قد يستهلك VRAM أكثر)")

            # 4. طباعة حالة نهائية واضحة
            try:
                final_device = next(self.pipeline.unet.parameters()).device
                final_dtype = next(self.pipeline.unet.parameters()).dtype
                print(f"  Pipeline جاهز على: {final_device}")
                print(f"  → نوع البيانات: {final_dtype}")
            except Exception as check_err:
                print(f"  !! فشل التحقق النهائي: {check_err}")

        else:
            raise RuntimeError("فشل تحميل Pipeline تمامًا – لا يمكن الاستمرار")

        # ──────────────────────────────────────────────────────────────
        # 4. تفعيل التحسينات لتوفير الذاكرة (فقط إذا كان cuda)
        # ──────────────────────────────────────────────────────────────
        if self.device.type == "cuda":
            print("  تفعيل تحسينات الـ VRAM...")
            try:
                pipe.enable_model_cpu_offload()          # ← هذا مهم
                print("  ✓ model_cpu_offload مفعّل")
            except Exception as e:
                print(f"  !! فشل cpu_offload: {e} → جاري تجربة sequential...")
                try:
                    pipe.enable_sequential_cpu_offload() # ← بديل أقوى في التوفير
                    print("  ✓ sequential_cpu_offload مفعّل")
                except Exception as seq_e:
                    print(f"  !! فشل sequential أيضًا: {seq_e}")

            # slicing – مفيد جدًا مع SDXL
            try:
                pipe.vae.enable_slicing()
                pipe.enable_attention_slicing("max")
                print("  ✓ VAE slicing + attention slicing مفعّل")
            except Exception as s_e:
                print(f"  !! فشل في تفعيل slicing: {s_e}")

        if self.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

        # هنا ممنوع تمامًا كتابة أي من الآتي:
        # pipe.to("cuda")
        # pipe.to(self.device)
        # pipe = pipe.to("cuda")   ← أي شكل من أشكال النقل اليدوي بعد offload

        # self.pipeline = pipe

        # ──────────────────────────────────────────────────────────────
        # 5. ربط الـ pipeline + إعدادات الأداء + رسالة نجاح
        # ──────────────────────────────────────────────────────────────
        if next(self.pipeline.unet.parameters()).device != self.device:
            print(f"→ نقل النموذج إلى {self.device}")
            self.pipeline = self.pipeline.to(self.device)

        # تفعيل تقنيات توفير الذاكرة إذا كان الجهاز cuda
        offload_status = "غير مفعّل"
        if self.device.type == "cuda":
            try:
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_model_cpu_offload()
                offload_status = "مفعّل (attention slicing + model CPU offload)"
                print("→ تفعيل attention slicing + model CPU offload")
            except Exception as e:
                offload_status = f"فشل تفعيل الـ offload/slicing: {str(e)}"
                print(f"  !! {offload_status}")
        else:
            offload_status = "غير مفعّل (CPU فقط)"

        # طباعة تقرير نهائي واضح ومفيد
        print("\n" + "═" * 70)
        print(" " * 20 + "✅ DNANetPulseRepair تم تهيئته بنجاح")
        print("═" * 70)
        print(f"   • الجهاز            : {self.device}")
        print(f"   • نوع البيانات      : {self.dtype}")
        print(f"   • ControlNet         : {self.controlnet.__class__.__name__}")
        print(f"   • Pipeline           : {self.pipeline.__class__.__name__}")
        print(f"   • حالة الـ Offload   : {offload_status}")

        # عرض استهلاك الذاكرة (اختياري لكن مفيد جدًا)
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   • الذاكرة المحجوزة تقريبًا : {reserved:.2f} GiB / {total:.2f} GiB")
        else:
            print("   • الذاكرة: غير متاح (CPU)")
        print("═" * 70 + "\n")

    # كشف المناطق "الميتة" أو المنهارة / منخفضة التفاصيل
    # ────────────────────────────────────────────────
    def detect_dead_zones(
            self,
            img: PILImage.Image,
            method: str = "multi",                  # "multi", "canny_dilate", "laplacian_var", "entropy"
            control_type: str = "tile",             # للتوافق مع استدعاءات أخرى (غير مستخدم حاليًا)
            canny_low: int = 60,
            canny_high: int = 180,
            dilation_kernel_size: int = 9,
            threshold: float = 0.25,
            min_area_ratio: float = 0.008,
            return_type: str = "mask"               # "mask", "score_map", "signed_map"
        ) -> PILImage.Image:
        print("→ detect_dead_zones بدأت | method=", method)
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
        # كل الـ scores لازم تكون في نطاق [0, 1] تقريبًا قبل الجمع
        # 0 = حي جدًا (جيد)، 1 = ميت جدًا (مشكلة)

        if method == "multi":
            # مزيج متوازن مع normalization بسيطة لكل score
            # نطبّع كل score لـ [0,1] لو كان خارج النطاق
            contrast_norm = np.clip(contrast_score, 0, 1)
            edge_norm     = np.clip(edge_density,    0, 1)
            sat_norm      = np.clip(sat_score,       0, 1)
            val_norm      = np.clip(val_score,       0, 1)

            # الأوزان (يمكن تمريرها كباراميتر لاحقًا)
            weights = {
                'contrast': 0.40,
                'edge':     0.30,
                'sat':      0.20,
                'val':      0.10
            }

            final_score = (
                weights['contrast'] * (1 - contrast_norm) +   # عكس: تباين عالي = حي
                weights['edge']     * (1 - edge_norm) +       # حواف كتير = حي
                weights['sat']      * sat_norm +              # تشبع منخفض = ميت
                weights['val']      * val_norm                # سطوع متطرف = ميت
            )

            # ضمان النطاق النهائي [0,1]
            final_score = np.clip(final_score, 0.0, 1.0)

        elif method == "canny_dilate":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            edge_density = dilated.astype(np.float32) / 255.0

            # عكس المنطق: كل ما الحواف أكتر = أقل ميت
            final_score = 1.0 - edge_density

            # إضافة threshold بسيط عشان نجنب الـ noise الصغير
            final_score[final_score < 0.05] = 0.0

        elif method == "laplacian_var":
            # تنفيذ حقيقي (بدل placeholder)
            # variance of Laplacian → مقياس للـ sharpness
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = cv2.variance(lap)  # scalar للصورة كلها (يمكن نافذة منزلقة لاحقًا)

            # لو variance عالي = حي، منخفض = ميت
            # normalization بسيطة (قيمة تجريبية، يمكن تعديلها)
            final_score = 1.0 - np.clip(lap_var / 1000.0, 0, 1)  # افتراضي

        elif method == "entropy":
            # تنفيذ بسيط لـ local entropy (أفضل من placeholder)
            # entropy عالي = تفاصيل كتير = حي
            from scipy.stats import entropy

            # لو عايز local entropy، استخدم نافذة منزلقة (بطيء شوية)
            # هنا نسخة global سريعة
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0,255), density=True)
            ent = entropy(hist + 1e-10)  # تجنب log(0)
            max_ent = np.log(256)        # أقصى entropy ممكن

            final_score = 1.0 - (ent / max_ent)  # entropy عالي = حي

        else:
            raise ValueError(f"طريقة غير مدعومة: {method}")

        # ──────────────── تحسين مشترك بعد الحساب ────────────────
        # تقليل الـ false positives في المناطق الداكنة جدًا أو الفاتحة
        if "val_score" in locals():
            extreme_val = (val_score > 0.95) | (val_score < 0.05)
            final_score[extreme_val] = np.maximum(final_score[extreme_val], 0.7)

        # smoothing خفيف عشان نجنب الـ noise الصغير
        final_score = cv2.GaussianBlur(final_score.astype(np.float32), (5,5), 1.0)

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
            # ماسك ثنائي (0 أو 255)
            return PILImage.fromarray((dead_mask > 0).astype(np.uint8) * 255).convert("L")

        elif return_type == "score_map":
            # normalization ديناميكية + تحويل لـ 0–255
            if final_score.size == 0:
                return PILImage.new("L", (1, 1), 128)  # fallback صغير

            min_val = final_score.min()
            max_val = final_score.max()
            if max_val == min_val:
                vis = np.full_like(final_score, 128, dtype=np.uint8)
            else:
                # نطاق آمن: نضيف هامش صغير لتجنب clipping قاسي
                range_val = max_val - min_val
                vis = ((final_score - min_val) / range_val * 255).clip(0, 255).astype(np.uint8)

            return PILImage.fromarray(vis).convert("L")

        elif return_type == "signed_map":
            # تدرج لوني حقيقي (أحمر → أصفر → أخضر)
            # -1.0 (ميت جدًا) → أحمر غامق
            #  0.0           → أصفر
            # +1.0 (حي جدًا) → أخضر فاتح

            # normalization إلى [-1, +1] تقريبًا
            score_norm = np.clip(final_score / 3.0, -1.0, 1.0)  # يمكن تعديل 3.0 حسب نطاقك

            # Hue: أحمر (0) → أصفر (30–60) → أخضر (85–120)
            hue = np.zeros_like(score_norm, dtype=np.float32)
            hue = np.where(score_norm <= 0,
                        0 + (score_norm + 1) * 30,          # -1 → 0, 0 → 30
                        30 + (score_norm) * 55)             # 0 → 30, +1 → 85

            sat = np.ones_like(score_norm) * 255
            val = np.ones_like(score_norm) * 255

            hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            return PILImage.fromarray(rgb)

        else:
            raise ValueError(f"return_type غير مدعوم: {return_type}")

    def _ensure_rgb(self, img: PILImage.Image) -> PILImage.Image:
        """
        التأكد من أن الصورة في وضع RGB
        """
        if img.mode != "RGB":
            return img.convert("RGB")
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

    def _create_canny_control(
        self,
        img_np: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
        blur_kernel_size: int = 5,          # جديد: لتقليل الضوضاء
        dilate_kernel_size: int = 3,
        dilate_iterations: int = 1,
    ) -> PILImage.Image:
        """
        إنشاء control image بنمط Canny مع preprocessing وdilation
        """
        # 1. التعامل مع القنوات (RGB أو RGBA أو grayscale)
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:  # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            elif img_np.shape[2] == 3:  # RGB
                pass
            else:
                raise ValueError(f"عدد القنوات غير مدعوم: {img_np.shape[2]}")
        elif len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"شكل الصورة غير مدعوم: {img_np.shape}")

        # 2. تحويل إلى grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 3. تقليل الضوضاء (اختياري لكن مفيد جدًا)
        if blur_kernel_size > 1:
            gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

        # 4. Canny edges
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # 5. توسيع (dilation) لربط الخطوط المتقطعة
        if dilate_kernel_size > 1:
            kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

        # 6. تحويل إلى RGB (ControlNet يفضل RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return PILImage.fromarray(edges_rgb)

    def _create_lineart_control(self, img_np: np.ndarray, thickness: int) -> PILImage.Image:
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

    def get_net_positive_prompt(self) -> str:
        return self.NET_POSITIVE_PROMPT

    def get_net_negative_prompt(self) -> str:
        return self.NET_NEGATIVE_PROMPT

    def _run_controlnet_inference(
        self,
        image: PILImage.Image,
        mask_image: PILImage.Image,
        control_image: PILImage.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance_scale: float,
    ) -> PILImage.Image:
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


    # ثوابت الـ prompts (مكانها الطبيعي في مستوى الكلاس)
    # ────────────────────────────────────────────────
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
        control_type: Literal["union", "tile", "canny", "lineart"] = "union",
        net_strength: float = 0.68,
        steps: int = 18,
        prompt: str = NET_POSITIVE_PROMPT,
        negative_prompt: str = NET_NEGATIVE_PROMPT,
        is_structure_only: bool = True,
    ) -> PILImage.Image:
        """
        توليد شبكة هيكلية موجهة بـ ControlNet للإصلاح الهندسي
        """
        img = img.convert("RGB") if img.mode != "RGB" else img
        mask = mask.convert("L")

        # ─── Control image preparation ───────────────────────────────
        control_img = img
        if control_type == "canny":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 180)
            control_img = PILImage.fromarray(edges).convert("RGB")

        elif control_type == "lineart":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            lineart = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            control_img = PILImage.fromarray(lineart).convert("RGB")

        # ─── Size validation ─────────────────────────────────────────
        if not (img.size == mask.size == control_img.size):
            raise ValueError(
                f"Size mismatch → img: {img.size}, mask: {mask.size}, control: {control_img.size}"
            )

        print(f"→ Generating structural net | control={control_type} | strength={net_strength:.2f} | steps={steps}")

        with torch.inference_mode(), torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=img,               # ← بدل image=
                mask_image=mask,              # ← لو مدعوم، وإلا شيله مؤقتًا
                control_image=control_img,
                strength=net_strength,
                num_inference_steps=steps,
                guidance_scale=7.2,
                controlnet_conditioning_scale=0.8,   # مهم
            ).images[0]

        print("→ Net structure generated")
        return result

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


    # ====================== 3. إعادة ترميم الشكل الهندسي ======================
    def repair_geometry_with_net(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        net: PILImage.Image,
        prompt: str = "high quality, detailed geometry, realistic structure, precise lines, architectural accuracy, masterpiece, best quality",
        negative_prompt: str = (
            "blurry, deformed, low detail, artifacts, noise, text, watermark, "
            "distorted geometry, bad anatomy, low quality, overexposed, underexposed, "
            "ugly, tiling, poorly drawn hands, mutation, jpeg artifacts, grainy"
        ),
        strength: float = 0.32,
        steps: int = 22,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.80,
        control_type: Optional[int] = None,
    ) -> PILImage.Image:
        """
        إعادة بناء الشكل الهندسي باستخدام الشبكة (net) كـ control
        متوافق مع StableDiffusionXLControlNetUnionPipeline
        """
        img = img.convert("RGB") if img.mode != "RGB" else img
        mask = mask.convert("L")
        net = net.convert("RGB") if net.mode != "RGB" else net

        if not (img.size == mask.size == net.size):
            raise ValueError(f"أحجام غير متطابقة: img={img.size}, mask={mask.size}, net={net.size}")

        print(f"→ repair_geometry | strength={strength:.2f} | steps={steps} | control_scale={controlnet_conditioning_scale:.2f}")

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": img,
            "mask_image": mask,
            "control_image": net,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        if control_type is not None:
            kwargs["control_type"] = control_type

        try:
            with torch.inference_mode(), torch.no_grad():
                output = self.pipeline(**kwargs)
                result = output.images[0]

            print("→ تم الترميم الهندسي بنجاح")
            return result

        except Exception as e:
            print(f"!! خطأ في repair_geometry_with_net: {type(e).__name__}: {str(e)}")
            # fallback بسيط: نرجع الصورة الأصلية عشان ما يوقفش الكل
            return img

    # ====================== 4. طبقات لونية مخصصة على أضلاع Net ======================
    def add_dna_colored_layers(
        self,
        net_image: PILImage.Image,
        mask: PILImage.Image,
        base_colors: List[Tuple[int, int, int]] = None,  # ← اجعلها اختيارية
        blend_mode: Literal["density", "wave", "genetic_random"] = "density",
        opacity: float = 0.50,
        use_random_colors: bool = True,  # ← الباراميتر الجديد
    ) -> PILImage.Image:
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
            opacity: الشفافية الأساسية للطبقة (0.0 → 1.0)

        Returns:
            طبقة RGBA جديدة يمكن دمجها فوق الصورة الأصلية بـ Image.alpha_composite أو blend
        """
        if base_colors is None:
            if self.color_engine:
                # استخدام المحرك إذا موجود
                fire = self.color_engine.generate_dnd_seed_color("Fire", variation=0.15)
                nature = self.color_engine.generate_dnd_seed_color("Nature", variation=0.12)
                ice = self.color_engine.generate_dnd_seed_color("Ice", variation=0.10)
                base_colors = [fire, nature, ice]
                print("→ تم توليد ألوان من D&D color engine")
            else:
                # fallback للألوان الثابتة
                base_colors = [(220, 60, 30), (40, 180, 60), (80, 180, 255)]
                print("→ استخدام ألوان افتراضية ثابتة")

            # تحضير الأساسيات مرة واحدة
            net_arr = np.array(net_image.convert("RGBA"))
            mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

            gray = cv2.cvtColor(net_arr[..., :3], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            h, w = mask_arr.shape
            layer = np.zeros((h, w, 4), dtype=np.float32)  # نعمل كل الحسابات على float32

            # ─── الدمج حسب الـ mode ─────────────────────────────────────────────
            if blend_mode == "density":
                intensity = np.clip(mask_arr ** 1.5, 0.0, 1.0)
                weights = np.stack([intensity * 0.60, intensity * 0.45, intensity * 0.35], axis=-1)
                weights = np.clip(weights, 0.0, 1.0)

                for i, color in enumerate(base_colors):
                    color_arr = np.array(color, dtype=np.float32)
                    layer[..., :3] += color_arr * weights[..., i, None]

            elif blend_mode == "wave":
                y, x = np.indices((h, w), dtype=np.float32)
                wave = np.sin(0.05 * y + 0.2 * x + np.linspace(0, 40 * np.pi, w))
                wave = (wave + 1) / 2
                wave = np.clip(wave, 0.0, 1.0)

                mixed = np.zeros((h, w, 3), dtype=np.float32)
                for i, color in enumerate(base_colors):
                    mixed += np.array(color, dtype=np.float32) * wave * (1.0 / len(base_colors))

                layer[..., :3] = mixed

            elif blend_mode == "genetic_random":
                ratios = np.random.dirichlet([1.0, 1.0, 1.0], size=(h, w))  # (h, w, 3)
                mixed = np.zeros((h, w, 3), dtype=np.float32)
                for i, color in enumerate(base_colors):
                    mixed += np.array(color, dtype=np.float32) * ratios[..., i, None]

                layer[..., :3] = mixed

            else:
                raise ValueError(f"blend_mode غير مدعوم: {blend_mode}")

            # ─── Alpha channel مرة واحدة ────────────────────────────────────────
            layer[..., 3] = mask_arr * 255 * opacity

            # ─── رسم الحواف (بعد الحسابات الرئيسية) ────────────────────────────
            layer[edges > 0, :3] = [220, 220, 255]
            layer[edges > 0, 3]   = 220

            # ─── تحويل نهائي آمن مرة واحدة فقط ────────────────────────────────
            layer = np.clip(np.round(layer), 0, 255).astype(np.uint8)

            return PILImage.fromarray(layer, mode="RGBA")


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

        result = PILImage.fromarray(rgb_out)
        print("  تم إرجاع الصورة الناتجة")
        print("="*60 + "\n")

        return result

    # ====================== الدالة الرئيسية (التنفيذ الكامل) ======================
    def repair(
        self,
        img: PILImage.Image,
        prompt: str = "masterpiece, best quality, highly detailed",
        use_colored_layers: bool = True,
        use_color_pulsing: bool = True,
        pulse_steps: int = 6,
        use_dna_zoom: bool = False,          # ← أضف هنا
        dna_zoom_factor: float = 1.5,        # اختياري: يمكن تمريره أو استخدام default
        dna_zoom_strength: float = 0.32,
        # ... باقي الباراميترات
    ) -> PILImage.Image:
        img = self._ensure_rgb(img)
        print("دخول repair() – بداية الدالة")

        # ───────────────────── 1. اكتشاف المناطق الميتة / الماسك ─────────────────────
        print("  قبل detect_dead_zones (الأولى)")
        mask = self.detect_dead_zones(img, threshold=0.35)
        if mask is None:
            print("تحذير: مفيش مناطق ميتة → نرجع الصورة كما هي")
            return img
        print(f"  بعد detect_dead_zones → mask: {mask is not None}, size={mask.size if mask else 'None'}")

        # ───────────────────── 2. توليد الشبكة الهيكلية (Net) ─────────────────────
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

        # ───────────────────── 3. طبقة DNA أساسية شفافة ─────────────────────
        print("  قبل create_dna_base_layer")
        dna_base = self.create_dna_base_layer(img.size, opacity=0.38)
        print(f"  بعد create_dna_base_layer → dna_base: {dna_base is not None}, size={dna_base.size if dna_base else 'None'}")

        # احتمال assert هنا (لو الكود بيتحقق من الطبقة)
        print("  بعد dna_base – هل في assert على dna_base؟")

        # ───────────────────── 4. الترميم الهندسي الرئيسي ─────────────────────
        print("  قبل repair_geometry_with_net")
        repaired = self.repair_geometry_with_net(
            img=img,
            mask=mask,
            net=net,
            prompt=prompt,          # اللي جاي من repair()
            strength=0.35,
            steps=25,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,   # ← جرب 0.6–1.0
        )
        print(f"  بعد repair_geometry_with_net → repaired: {repaired is not None}, size={repaired.size if repaired else 'None'}")

        # احتمال assert هنا (غالباً ده المكان المشتبه فيه)
        print("  بعد repaired – هل في assert على repaired؟ (لو ما ظهرش اللي بعده يبقى فيه assert هنا)")

        # ───────────────────── 5. طبقات لونية (اختياري) ─────────────────────
        if use_colored_layers:
            print("  قبل add_dna_colored_layers")
            colored = self.add_dna_colored_layers(
                net_image=net,
                mask=mask,
                base_colors=[(220, 60, 30), (40, 180, 60), (80, 180, 255)],   # ← غيّر base_colors إلى colors
                blend_mode="density",
                opacity=0.55,
            )
            print(f"  بعد add_dna_colored_layers → colored: {colored is not None}, size={colored.size if colored else 'None'}")

            print("  قبل alpha_composite")
            repaired = Image.alpha_composite(repaired.convert("RGBA"), colored)
            print("  بعد alpha_composite → repaired تم تعديله")
        else:
            print("  use_colored_layers = False → تم تخطي الطبقات اللونية")

        # ───────────────────── 6. نبض لوني DNA-inspired (اختياري) ─────────────────────
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

        # ───────────────────── 7. ماسك ثانٍ + تلميع نهائي ─────────────────────
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

        # ───────────────────── 8 داخل repair() بعد generate_net_structure ─────────────────────
        print("قبل repair_geometry_with_net")
        repaired = self.repair_geometry_with_net(img=img, mask=mask, net=net, strength=0.30, steps=25)
        print("بعد repair_geometry_with_net →", repaired.size if repaired else "فشل")

        print("قبل dna_zoom_repair")
        repaired = self.dna_zoom_repair(
            img=repaired,
            mask=mask,
            net=net,
            zoom_factor=1.5,
            strength=0.32,
            controlnet_conditioning_scale=0.85,
            steps=28,
        )
        print("بعد dna_zoom_repair →", repaired.size if repaired else "فشل")

        print("  قبل التلميع النهائي")
        final = ImageEnhance.Sharpness(final).enhance(1.10)
        final = ImageEnhance.Contrast(final).enhance(1.05)
        print("  بعد التلميع النهائي")

        print("  خروج repair() – نجاح (لو وصلنا هنا يبقى مفيش assert وقّفنا)")
        return final.convert("RGB")


    def dna_zoom_repair(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        net: PILImage.Image,
        zoom_factor: float = 1.5,
        prompt: str = "...",
        negative_prompt: str = "...",
        strength: float = 0.32,
        steps: int = 28,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.85,
        control_type: Optional[int] = None,
    ) -> PILImage.Image:
        """
        إعادة بناء / تحسين منطقة معينة بأسلوب DNA.Zoom:
        1. تكبير المنطقة المقنعة (zoom)
        2. توليد تفاصيل أعلى باستخدام الـ net كـ control
        3. إعادة وضع النتيجة في الصورة الأصلية
        """
        from PIL import Image, ImageOps

        # التأكد من الأنماط
        img   = img.convert("RGB")
        mask  = mask.convert("L")
        net   = net.convert("RGB")

        if img.size != mask.size or img.size != net.size:
            raise ValueError("يجب أن تكون أحجام الصورة + الماسك + الـ net متساوية")

        # ─── 1. استخراج المنطقة المقنعة + تكبيرها ──────────────────────────────
        bbox = mask.getbbox()
        if bbox is None:
            print("تحذير: الماسك فارغ → نرجع الصورة كما هي")
            return img

        x1, y1, x2, y2 = bbox
        cropped_img  = img.crop(bbox)
        cropped_mask = mask.crop(bbox)
        cropped_net  = net.crop(bbox)

        # حساب الحجم الجديد بعد الزوم
        new_w = int(cropped_img.width  * zoom_factor)
        new_h = int(cropped_img.height * zoom_factor)

        zoomed_img  = cropped_img.resize((new_w, new_h), Image.LANCZOS)
        zoomed_mask = cropped_mask.resize((new_w, new_h), Image.NEAREST)
        zoomed_net  = cropped_net.resize((new_w, new_h), Image.LANCZOS)

        # توسيع الماسك قليلاً لتجنب الحواف القاسية بعد الـ paste
        zoomed_mask = ImageOps.expand(zoomed_mask, border=8, fill=0)
        zoomed_mask = zoomed_mask.resize((new_w, new_h), Image.NEAREST)

        # ─── 2. التوليد داخل المنطقة المكبرة ────────────────────────────────
        print(f"DNA.Zoom → zoom={zoom_factor:.2f} | strength={strength:.2f} | control_scale={controlnet_conditioning_scale:.2f} | steps={steps}")

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": zoomed_img,
            "mask_image": zoomed_mask,
            "control_image": zoomed_net,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        if control_type is not None:
            kwargs["control_type"] = control_type

        with torch.inference_mode(), torch.no_grad():
            output = self.pipeline(**kwargs)
            generated = output.images[0]

        # إعادة تصغير النتيجة للحجم الأصلي للمنطقة المقصوصة
        generated = generated.resize((x2 - x1, y2 - y1), Image.LANCZOS)

        # ─── 3. لصق النتيجة في الصورة الأصلية ────────────────────────────────
        result = img.copy()
        result.paste(generated, (x1, y1), generated)   # لو كان في alpha → استخدم mask

        # اختياري: تحسين الحواف بـ feathering بسيط
        if zoom_factor > 1.2:
            from PIL import ImageFilter
            result = result.filter(ImageFilter.GaussianBlur(radius=0.8))

        print("DNA.Zoom انتهى")
        return result


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
