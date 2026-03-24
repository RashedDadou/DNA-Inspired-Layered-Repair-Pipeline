# union_multi_inpainting.py (نسخة محسنة ومنظمة 2026)

from __future__ import annotations

import sys
import os
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image as PILImage, ImageFilter
from typing import Union, Optional, List, Any, Dict, Tuple

import torch
from diffusers.pipelines.controlnet.pipeline_controlnet_union_sd_xl_img2img import StableDiffusionXLControlNetUnionImg2ImgPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel

from tqdm import tqdm

try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# controlnet_aux (اختياري) - يتم استيراده داخل الدالة لتجنب تحذيرات Pylance
OpenposeDetector = None
ZoeDetector = None
HWC3 = None

try:
    from controlnet_aux import OpenposeDetector, ZoeDetector
    from controlnet_aux.util import HWC3   # ← هذي الأصلية والأسرع
except ImportError:
    pass

try:
    import piexif
except ImportError:
    piexif = None

# ─── Logging ───────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── CONTROL TYPE MAP (ثابت مهم) ───────────────────────────────────────────────────
CONTROL_TYPE_MAP = {
    "canny":     0,
    "tile":      1,
    "depth":     2,
    "lineart":   3,
    "openpose":  4,
    "scribble":  5,
    "hed":       6,
    "mlsd":      7,
    "seg":       8,
    "normal":    9,
    "softedge": 10,
}

# ────────────────────────────────────────────────
#   كلاس الإداري Pipeline Manager
# ────────────────────────────────────────────────
class PipelineManager:
    """
    Singleton + Single Source of Truth لتحميل الـ Union Pipeline
    """
    _instance = None

    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_DTYPE = torch.float16
    DEFAULT_LOCAL_CONTROLNET = "./models/controlnet-union-sdxl-1.0"

    # النسخة الصحيحة والمستقرة
    DEFAULT_HF_CONTROLNET = "xinsir/controlnet-union-sdxl-1.0"

    DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipe = None
            cls._instance.device = cls.DEFAULT_DEVICE
            cls._instance.dtype = cls.DEFAULT_DTYPE
        return cls._instance

    @lru_cache(maxsize=1)
    def get_pipeline(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        sdxl_model_id: Optional[str] = None,
        low_vram_mode: str = "balanced",          # off / balanced / very_low / extreme
        force_reload: bool = False,
        compile_unet: bool = False,
        enable_xformers: bool = True,
        local_controlnet_path: Optional[str] = None,
        hf_controlnet_id: Optional[str] = None,
    ) -> StableDiffusionXLControlNetUnionImg2ImgPipeline:
        """
        تحميل الـ SDXL Union Pipeline (Single Source of Truth)
        مع دعم كامل لـ VRAM modes + xformers + torch.compile + fuse_qkv
        """
        dtype = dtype or getattr(self, 'dtype', torch.float16)
        device = torch.device(device or getattr(self, 'device', "cuda" if torch.cuda.is_available() else "cpu"))

        if force_reload or self._pipe is None:
            if force_reload:
                self.get_pipeline.cache_clear()
                logger.info("Cache cleared → forced reload")

            logger.info(f"🔄 تحميل Pipeline جديدة | {device} | {dtype} | mode={low_vram_mode} | compile={compile_unet}")

            try:
                scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                    self.DEFAULT_SDXL_MODEL, subfolder="scheduler"
                )

                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
                )

                # ─── ControlNet: local → HF ───
                controlnet_path = local_controlnet_path or self.DEFAULT_LOCAL_CONTROLNET
                controlnet = None

                try:
                    if os.path.exists(controlnet_path):
                        controlnet = ControlNetUnionModel.from_pretrained(
                            controlnet_path,
                            torch_dtype=dtype,
                            use_safetensors=True,
                            local_files_only=True
                        )
                        logger.info("✓ ControlNet Union محمل من المسار المحلي")
                    else:
                        raise FileNotFoundError
                except Exception:
                    hf_id = hf_controlnet_id or self.DEFAULT_HF_CONTROLNET

                    # الحل المهم: لا نستخدم variant="fp16" مع هذا الـ model
                    controlnet = ControlNetUnionModel.from_pretrained(
                        hf_id,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        # variant="fp16"   ← احذف هذا السطر أو اجعله None
                    )
                    logger.info(f"✓ ControlNet Union محمل من Hugging Face: {hf_id}")

                # ─── Model ID (أولوية: parameter > env > default) ───
                model_id = sdxl_model_id or os.getenv("SDXL_MODEL_ID") or self.DEFAULT_SDXL_MODEL

                pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    vae=vae,
                    scheduler=scheduler,
                    torch_dtype=dtype,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                )

                # ─── VRAM Management ───
                if low_vram_mode in ("very_low", "extreme"):
                    pipe.enable_sequential_cpu_offload()
                elif low_vram_mode == "balanced":
                    pipe.enable_model_cpu_offload()

                if low_vram_mode != "off":
                    slice_size = "max" if low_vram_mode == "extreme" else "auto"
                    pipe.enable_attention_slicing(slice_size)

                if low_vram_mode in ("very_low", "extreme"):
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()

                # 🔥 fuse_qkv_projections (مهمة جدًا في extreme mode)
                if low_vram_mode == "extreme":
                    try:
                        pipe.fuse_qkv_projections()
                        logger.info("✓ Enabled fuse_qkv_projections")
                    except AttributeError:
                        logger.warning("fuse_qkv_projections غير متوفر في هذا الإصدار")

                pipe.to(device)

                # ─── xformers + torch.compile ───
                if enable_xformers and device.type == "cuda":
                    try:
                        import xformers
                        pipe.enable_xformers_memory_efficient_attention()
                        logger.info("✓ xformers مفعّل")
                    except Exception:
                        pass

                if compile_unet and torch.__version__ >= "2.0" and device.type == "cuda":
                    try:
                        pipe.unet = torch.compile(
                            pipe.unet,
                            mode="reduce-overhead",
                            fullgraph=True,
                            dynamic=True
                        )
                        logger.info("✓ UNet compiled successfully")
                    except Exception as e:
                        logger.warning(f"torch.compile فشل: {e}")

                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = False

                self._pipe = pipe
                logger.info("✅ Pipeline جاهزة تمامًا | device=%s | dtype=%s", pipe.device, dtype)

            except Exception as e:
                logger.critical("فشل تحميل الـ Pipeline", exc_info=True)
                raise RuntimeError(f"تعذر تحميل الـ pipeline: {str(e)}") from e

        return self._pipe

    def set_defaults(self, device=None, dtype=None):
        """تغيير الافتراضيات بعد الإنشاء"""
        if device is not None:
            self.device = device
            logger.info(f"تم تغيير device إلى {device}")
        if dtype is not None:
            self.dtype = dtype
            logger.info(f"تم تغيير dtype إلى {dtype}")

    def clear_cache(self):
        self.get_pipeline.cache_clear()
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ─── دوال مستقلة ───────────────────────────────────────────────────────────────
def blend_with_mask(
    generated: PILImage.Image,
    original: PILImage.Image,
    mask: Optional[PILImage.Image],
    edge_feather_radius: int = 16,
    use_poisson_blend: bool = True,
    poisson_mode: str = "normal",
    fallback_to_feather_if_poisson_fails: bool = True,
) -> PILImage.Image:
    """
    دمج الصورة المولدة مع الأصلية باستخدام الماسك.
    تدعم Poisson Blending (أفضل جودة) مع fallback إلى Gaussian Feather.
    """
    if mask is None:
        return generated

    if original is None:
        logger.warning("original image is None → returning generated image")
        return generated

    try:
        # التأكد من نفس الحجم
        target_size = original.size
        generated = generated.resize(target_size, PILImage.LANCZOS)
        mask = mask.convert("L").resize(target_size, PILImage.NEAREST)

        # تحويل إلى numpy
        gen_np = np.array(generated.convert("RGB"))
        orig_np = np.array(original.convert("RGB"))
        mask_np = np.array(mask).astype(np.float32) / 255.0

        # ─── Poisson Blending (الطريقة الأفضل) ───
        if use_poisson_blend:
            try:
                # حساب مركز المنطقة الماسكة
                mask_binary = (mask_np > 0.05).astype(np.uint8)
                ys, xs = np.nonzero(mask_binary)

                if len(xs) > 0 and len(ys) > 0:
                    center = (int(np.mean(xs)), int(np.mean(ys)))
                else:
                    center = (gen_np.shape[1] // 2, gen_np.shape[0] // 2)
                    logger.debug("لم يتم العثور على منطقة ماسك → استخدام المنتصف")

                # اختيار وضع الدمج
                clone_mode = cv2.NORMAL_CLONE
                if poisson_mode == "mixed":
                    clone_mode = cv2.MIXED_CLONE
                elif poisson_mode not in ("normal", "mixed"):
                    logger.warning(f"poisson_mode غير معروف '{poisson_mode}' → استخدام NORMAL_CLONE")

                blended = cv2.seamlessClone(
                    gen_np,
                    orig_np,
                    (mask_np * 255).astype(np.uint8),
                    center,
                    clone_mode
                )
                return PILImage.fromarray(blended)

            except Exception as poisson_err:
                logger.warning(f"Poisson blending فشل: {poisson_err}")
                if not fallback_to_feather_if_poisson_fails:
                    raise

        # ─── Fallback: Gaussian Feather ───
        feather = edge_feather_radius
        if feather <= 0:
            blended = (
                mask_np[..., None] * gen_np +
                (1 - mask_np[..., None]) * orig_np
            ).astype(np.uint8)
        else:
            ksize = feather * 2 + 1
            mask_blur = cv2.GaussianBlur(
                mask_np,
                (ksize, ksize),
                sigmaX=feather
            )
            mask_blur = np.clip(mask_blur, 0.0, 1.0)

            blended = (
                mask_blur[..., None] * gen_np +
                (1 - mask_blur[..., None]) * orig_np
            ).astype(np.uint8)

        return PILImage.fromarray(blended)

    except Exception as e:
        logger.error(f"فشل دمج الماسك بالكامل: {type(e).__name__}: {e}", exc_info=True)
        # Fallback أخير: نرجع الصورة المولدة كما هي
        return generated

def normalize_controls(
    controls: List[Tuple[PILImage.Image, int, float]],
    logger: Optional[logging.Logger] = None
) -> Tuple[List[PILImage.Image], List[int], List[float]]:
    """تطبيع الـ controls وإرجاع ثلاث قوائم منفصلة (النسخة الأساسية النظيفة)"""
    if not controls:
        return [], [], []

    images: List[PILImage.Image] = []
    types: List[int] = []
    scales: List[float] = []

    for img, typ, scl in controls:
        if not isinstance(img, PILImage.Image):
            if logger:
                logger.warning("عنصر غير صورة في controls → تم تجاهله")
            continue

        images.append(img)

        try:
            types.append(int(typ))
        except (TypeError, ValueError):
            types.append(4)  # fallback openpose
            if logger:
                logger.warning(f"control_type غير صالح ({typ}) → استخدام 4 (openpose)")

        try:
            scales.append(float(scl))
        except (TypeError, ValueError):
            scales.append(0.75)
            if logger:
                logger.warning(f"control_scale غير صالح ({scl}) → استخدام 0.75")

    return images, types, scales

def prepare_control_kwargs(
    controls: List[Tuple[PILImage.Image, int, float]],
    use_union: bool = True,
    strict: bool = False,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
) -> Dict[str, Any]:
    """تحضير kwargs لـ ControlNet Union SDXL - تدعم multi-control كامل"""
    if not controls:
        return {}

    images, control_types, control_scales = normalize_controls(controls, logger=logger)

    n = len(images)
    if n == 0:
        logger.warning("لم يتبقَ أي control image صالحة")
        return {}

    if len(control_types) != n or len(control_scales) != n:
        msg = f"أطوال غير متسقة (images={n}, types={len(control_types)}, scales={len(control_scales)})"
        if strict:
            raise ValueError(msg)
        logger.error(msg)

    if n == 1:
        return {
            "union_control": use_union,
            "control_image": images[0],
            "control_mode": control_types[0],
            "controlnet_conditioning_scale": control_scales[0],
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
        }

    # Multi-Control (n > 1)
    return {
        "union_control": use_union,
        "control_image": images,
        "control_mode": control_types,
        "controlnet_conditioning_scale": control_scales,
        "control_guidance_start": [control_guidance_start] * n,
        "control_guidance_end": [control_guidance_end] * n,
    }

# ────────────────────────────────────────────────
#   UnionGenerator (مُنظّف)
# ────────────────────────────────────────────────
class UnionGenerator:
    """
    الكلاس الرئيسي لتوليد الصور باستخدام SDXL + ControlNet Union + Inpainting
    """

    def __init__(self, args):
        """تهيئة الكلاس وتحميل الإعدادات الأساسية"""
        self.args = args
        self.manager = PipelineManager()

        self.device = self.manager.device
        self.dtype = self.manager.dtype
        self.seed = self._get_seed()

        self.pipe: Optional[StableDiffusionXLControlNetUnionImg2ImgPipeline] = None
        self.source_img: Optional[PILImage.Image] = None
        self.mask_img: Optional[PILImage.Image] = None

        self.controls: List[Tuple[PILImage.Image, int, float]] = []
        self.user_controls: List[Tuple[PILImage.Image, int, float]] = []

        self.controls_prepared = False
        self.controls_disabled = False
        self.controls_disabled_reason: str = ""
        self.use_union = False

        # تحميل الـ detectors قبل إنشاء الـ logger
        self.openpose_detector = None
        self.zoe_detector = None
        self._load_detectors()

        # إنشاء الـ logger بعد تحميل الـ detectors
        self.logger = self._setup_logging()
        self.logger.info(f"UnionGenerator تم تهيئته | device={self.device} | seed={self.seed}")

    def HWC3(self, x: np.ndarray) -> np.ndarray:
        """
        تحويل صورة من HWC إلى CHW أو العكس (مساعد من controlnet_aux)
        تستخدم عادة لمعالجة خرائط الـ ControlNet.
        """
        if x is None:
            return None

        if len(x.shape) == 3 and x.shape[2] == 3:  # HWC → CHW
            return x.transpose(2, 0, 1).copy()
        elif len(x.shape) == 3 and x.shape[0] == 3:  # CHW → HWC
            return x.transpose(1, 2, 0).copy()

        return x  # إذا كانت الصورة بالفعل في الشكل الصحيح

    # ====================== دوال التحميل Load ======================
    def _load_detectors(self):
        """تحميل OpenposeDetector و ZoeDetector مرة واحدة فقط"""
        if not getattr(self.args, 'auto_controls', True):
            return

        try:
            from controlnet_aux import OpenposeDetector, ZoeDetector

            print("جاري تحميل OpenposeDetector و ZoeDetector...")   # استخدم print مؤقتاً

            self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.zoe_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators")

            print("✅ تم تحميل Openpose و Zoe Detector بنجاح (مرة واحدة)")

        except Exception as e:
            print(f"⚠️ فشل تحميل الـ detectors: {e}")
            self.openpose_detector = None
            self.zoe_detector = None

    def _setup_logging(self):
        """إعداد الـ logger الخاص بالكلاس"""
        logger = logging.getLogger("UnionSDXL")
        logger.setLevel(logging.DEBUG if getattr(self.args, 'debug', False) else logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(funcName)s | %(message)s"
        ))

        # تجنب إضافة handler متعددة
        if not logger.handlers:
            logger.addHandler(handler)

        return logger

    def _get_seed(self) -> int:
        """إرجاع الـ seed من الـ args أو توليد عشوائي"""
        if self.args.seed >= 0:
            return self.args.seed
        return random.randint(0, 2**31 - 1)

    def _mark_controls_disabled(self, reason: str):
        """تسجيل أن الـ controls معطلة + حفظ السبب للاستخدام لاحقاً (metadata / اسم الملف)"""
        self.controls_disabled = True
        self.controls_disabled_reason = reason
        self.logger.warning(f"Controls معطلة: {reason}")

    # ====================== دوال التحميل ======================
    def load_image_and_mask(self):
        """تحميل الصورة الأساسية + الماسك (اختياري) مع تحققات أمان كاملة"""
        input_path = Path(self.args.input)

        # 1. التحقق من وجود الصورة الأساسية
        if not input_path.is_file():
            self.logger.critical(f"الصورة الأساسية غير موجودة: {input_path}")
            sys.exit(1)

        try:
            self.source_img = PILImage.open(input_path).convert("RGB")
            self.logger.info(f"تم تحميل الصورة الأساسية: {input_path}  ({self.source_img.size})")
        except Exception as e:
            self.logger.critical("فشل فتح الصورة الأساسية", exc_info=True)
            sys.exit(1)

        # 2. تحميل الماسك (اختياري)
        self.mask_img: Optional[PILImage.Image] = None
        if getattr(self.args, 'mask', None):
            mask_path = Path(self.args.mask)
            if mask_path.is_file():
                try:
                    self.mask_img = PILImage.open(mask_path).convert("L")
                    self.logger.info(f"تم تحميل الماسك بنجاح: {mask_path}")
                except Exception as e:
                    self.logger.warning(f"فشل فتح الماسك: {e} → سيتم الاستمرار بدون ماسك")
                    self.mask_img = None
            else:
                self.logger.warning(f"مسار الماسك غير صالح: {mask_path} → سيتم تجاهله")
        else:
            self.logger.info("ماسك غير محدد → سيتم التشغيل بدون ماسك")

    def _load_user_controls(self):
        """تحميل control images من arguments مع دعم كامل"""
        control_map = {
            "control_openpose":  (4, 1.00),   # openpose
            "control_depth":     (2, 0.85),   # depth
            "control_canny":     (0, 0.80),   # canny
            "control_tile":      (1, 0.60),   # tile
            "control_scribble":  (5, 0.90),   # scribble
            "control_hed":       (6, 0.85),   # hed
            "control_softedge":  (10, 0.85),  # softedge
            "control_lineart":   (3, 0.80),   # lineart
            "control_mlsd":      (7, 0.75),   # mlsd
            "control_normal":    (9, 0.80),   # normal
            "control_seg":       (8, 0.70),   # seg
        }

        for arg_name, (typ, default_scale) in control_map.items():
            # تحويل snake_case إلى الاسم في argparse
            attr_name = arg_name.replace("_", "-")
            path = getattr(self.args, arg_name, None)

            if path and Path(path).is_file():
                try:
                    img = PILImage.open(path).convert("RGB")
                    self.user_controls.append((img, typ, default_scale))
                    self.logger.info(f"✅ تم تحميل control: --{attr_name} → type {typ}")
                except Exception as e:
                    self.logger.warning(f"فشل تحميل --{attr_name}: {e}")

    def prepare_controls(self):
        """
        تحضير الـ controls:
        - يدعم controls من المستخدم (user_controls) بأولوية عالية
        - يدعم الـ auto controls (OpenPose + Depth) حسب الإعداد
        """
        if self.controls_prepared:
            self.logger.debug("Controls سبق وتم تحضيرها → تم تخطي")
            return

        if self.source_img is None:
            self.logger.error("لا توجد صورة محملة لإنشاء الـ controls")
            self._mark_controls_disabled("لا توجد صورة مصدر")
            self.controls_prepared = True
            return

        self.controls = []   # إعادة تهيئة

        # 1. إضافة الـ controls الخارجية من المستخدم (أولوية أولى)
        if self.user_controls:
            self.controls.extend(self.user_controls)
            self.logger.info(f"✅ تم إضافة {len(self.user_controls)} control map(s) من المستخدم")

        # 2. الـ controls التلقائية (OpenPose + Depth)
        if getattr(self.args, 'auto_controls', True):
            try:
                from controlnet_aux import OpenposeDetector, ZoeDetector
            except ImportError:
                self.logger.error("مكتبة controlnet_aux غير مثبتة!")
                self._mark_controls_disabled("controlnet_aux غير مثبتة")
                self.controls_prepared = True
                return

            # OpenPose
            try:
                pose_proc = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                pose_map = pose_proc(self.source_img)
                self.controls.append((pose_map, 4, 1.00))
                self.logger.info("✅ تم إنشاء OpenPose map تلقائياً")
            except Exception as e:
                self.logger.warning(f"فشل إنشاء OpenPose التلقائي: {type(e).__name__} - {e}")

            # Zoe Depth
            try:
                depth_proc = ZoeDetector.from_pretrained("lllyasviel/Annotators")
                depth_map = depth_proc(self.source_img)
                self.controls.append((depth_map, 2, 0.85))
                self.logger.info("✅ تم إنشاء Zoe Depth map تلقائياً")
            except Exception as e:
                self.logger.warning(f"فشل إنشاء Zoe Depth التلقائي: {type(e).__name__} - {e}")

        # التحقق النهائي
        self.controls_prepared = True

        if not self.controls:
            msg = "فشل إنشاء أي control maps"
            if hasattr(self, 'controls_disabled_reason') and self.controls_disabled_reason:
                msg += f" ({self.controls_disabled_reason})"
            self.logger.critical(msg)
            self._mark_controls_disabled(msg)
            if getattr(self.args, 'strict_mode', False):
                raise RuntimeError(msg)
        else:
            auto_count = len(self.controls) - len(self.user_controls)
            self.logger.info(f"✅ إجمالي controls جاهزة: {len(self.controls)} "
                            f"(من المستخدم: {len(self.user_controls)} | تلقائي: {auto_count})")

    def ensure_pipe(self):
        """تأكد من تحميل الـ pipeline عبر PipelineManager (Single Source of Truth)"""
        if getattr(self, 'pipe', None) is None:
            self.logger.info("الـ pipeline غير محملة → جاري التحميل الآن عبر PipelineManager...")

            try:
                self.pipe = self.manager.get_pipeline(
                    dtype=self.dtype,
                    device=self.device,
                    low_vram_mode=getattr(self.args, 'vram_mode', 'balanced'),
                    force_reload=False,
                    compile_unet=getattr(self.args, 'compile_unet', False),
                    enable_xformers=True,
                )
                self.logger.info("✅ Pipeline محملة بنجاح داخل الكلاس")
            except Exception as e:
                self.logger.critical("فشل تحميل الـ pipeline", exc_info=True)
                raise
        else:
            self.logger.debug("الـ pipeline موجودة بالفعل")

    # ====================== الدالة الرئيسية للتوليد ======================
    def generate(self):
        """
        الدالة الرئيسية للتوليد (img2img + Union ControlNet + ماسك blending)
        """
        self.logger.info("🚀 بدء عملية التوليد...")

        # ====================== 1. التأكد من تحميل الـ Pipeline وإعداد VRAM ======================
        self.ensure_pipe()

        if getattr(self.args, 'vram_mode', 'balanced') != "off":
            self.pipe.to("cpu")
            self.logger.debug("Pipeline moved back to CPU due to offloading mode")

        # ====================== 2. حساب قرار استخدام Union + التحققات ======================
        use_union = getattr(self.args, 'use_union', True) and bool(self.controls) and not self.controls_disabled

        if self.use_union and not self.controls:
            if self.controls_disabled:
                self.logger.warning(f"Union مطلوب لكن الـ controls معطلة: {self.controls_disabled_reason}")
            else:
                raise ValueError("use_union=True لكن لا توجد control images!")

        if self.pipe is None:
            raise RuntimeError("فشل تحميل الـ Pipeline قبل بدء التوليد!")

        self.logger.info(f"Union مفعّل: {self.use_union} | Controls: {len(self.controls)} | Disabled: {self.controls_disabled}")

        # ====================== 3. التحجيم المسبق للصورة والـ Controls ======================
        target_size = self.get_target_size(self.source_img) if self.source_img is not None else (1024, 1024)
        source_resized = self.safe_resize(self.source_img, target_size)

        if self.use_union and len(self.controls) > 0:
            self.controls = [
                (self.resize_control_to_match(img, target_size), typ, scl)
                for img, typ, scl in self.controls
            ]

        # ====================== 4. بناء الـ pipe_kwargs الأساسية ======================
        pipe_kwargs: Dict[str, Any] = {
            "prompt": self.args.prompt,
            "negative_prompt": self.args.negative or "",
            "strength": self.args.strength,
            "num_inference_steps": self.args.steps,
            "guidance_scale": self.args.cfg,
            "generator": torch.Generator(device=self.device).manual_seed(self.seed),
            "return_dict": True,
        }

        # ====================== 5. إضافة ControlNet Union (الجزء الحساس) ======================
        if self.use_union and len(self.controls) > 0:
            num_controls = len(self.controls)

            control_kwargs = prepare_control_kwargs(
                controls=self.controls,
                use_union=True,
                strict=getattr(self.args, 'strict_mode', False),
                control_guidance_start=getattr(self.args, 'control_start', 0.0),
                control_guidance_end=getattr(self.args, 'control_end', 1.0),
            )

            pipe_kwargs["image"] = source_resized

            pipe_kwargs.update({
                "control_image": control_kwargs["control_image"],
                "control_mode": control_kwargs["control_mode"],
                "controlnet_conditioning_scale": control_kwargs["controlnet_conditioning_scale"],
                "control_guidance_start": control_kwargs["control_guidance_start"],
                "control_guidance_end": control_kwargs["control_guidance_end"],
            })

            self.logger.info(f"✅ Union Multi-Control مفعّل مع {num_controls} control(s)")

        else:
            pipe_kwargs["image"] = source_resized
            self.logger.info("⚠️ Union غير مفعّل → img2img عادي")

        # ====================== 6. إعداد Callback للتقدم (tqdm) ======================
        class TqdmCallback:
            def __init__(self, total: int):
                self.pbar = tqdm(total=total, desc="Sampling", unit="step", leave=False)

            def __call__(self, pipe, step: int, timestep: int, callback_kwargs):
                self.pbar.update(1)
                return callback_kwargs

            def close(self):
                self.pbar.close()

        callback_handler = TqdmCallback(self.args.steps)
        pipe_kwargs["callback_on_step_end"] = callback_handler

        # ====================== 7. الـ Inference الفعلي ======================
        try:
            output = self.pipe(**pipe_kwargs)

            # استخراج الصورة بأمان مع التعامل مع كل الأنواع الممكنة من الـ output
            generated = None

            if hasattr(output, "images") and output.images is not None:
                # الحالة الطبيعية: return_dict=True
                if isinstance(output.images, (list, tuple)) and len(output.images) > 0:
                    generated = self.ensure_pil_image(output.images[0])
                else:
                    generated = self.ensure_pil_image(output.images)

            elif isinstance(output, (list, tuple)):
                # الحالة اللي بترجع list أو tuple مباشرة
                first_item = output[0]
                if isinstance(first_item, (list, tuple)) and len(first_item) > 0:
                    generated = self.ensure_pil_image(first_item[0])
                else:
                    generated = self.ensure_pil_image(first_item)

            # fallback آمن
            if generated is None:
                self.logger.warning("مخرج الـ pipeline غير متوقع → استخدام source_resized كـ fallback")
                generated = source_resized

        except Exception as e:
            self.logger.error(f"❌ فشل الـ inference: {type(e).__name__}", exc_info=True)
            generated = source_resized

        finally:
            callback_handler.close()

        # ====================== 8. دمج الماسك (Blending) ======================
        if self.mask_img is not None and self.source_img is not None:
            try:
                self.result = blend_with_mask(
                    generated=generated,
                    original=self.source_img,
                    mask=self.mask_img,
                    edge_feather_radius=getattr(self.args, 'feather_radius', 16),
                    use_poisson_blend=getattr(self.args, 'use_poisson', True)
                )
                self.logger.info("✅ تم دمج الماسك بنجاح")
            except Exception as e:
                self.logger.warning(f"⚠ فشل دمج الماسك: {e}")
                self.result = generated
        else:
            self.result = generated

        self.logger.info(f"🎉 انتهى التوليد بنجاح | seed={self.seed}")

    def save(self):
        """حفظ الصورة مع تجنب التكرار + إضافة EXIF metadata كامل"""
        if not hasattr(self, 'result') or self.result is None:
            self.logger.error("لا توجد صورة مولدة للحفظ!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_stem = f"union_{timestamp}_s{self.seed}"
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{base_stem}.png"

        # تجنب تكرار اسم الملف
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{base_stem}_{counter:03d}.png"
            counter += 1

        # حفظ الصورة
        self.result.save(output_path, quality=95, optimize=True)
        self.logger.info(f"تم حفظ الصورة: {output_path}")

        # إضافة EXIF Metadata
        try:
            if piexif is None:
                self.logger.warning("piexif غير مثبتة → بدون EXIF")
                return

            try:
                exif_dict = piexif.load(str(output_path))
            except:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            description_lines = [
                f"Prompt: {self.args.prompt}",
                f"Negative: {self.args.negative or ''}",
                f"Seed: {self.seed}",
                f"Steps: {self.args.steps}",
                f"CFG: {self.args.cfg:.2f}",
                f"Strength: {self.args.strength:.2f}",
                f"Model: SDXL + ControlNet Union",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Device: {getattr(self.pipe, 'device', 'unknown')}",
            ]

            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    description_lines.append(f"VRAM: ~{allocated:.1f}GB alloc / ~{reserved:.1f}GB reserved")
                except:
                    pass

            if self.result:
                w, h = self.result.size
                description_lines.append(f"Resolution: {w}×{h}")

            if self.controls:
                description_lines.append(f"Controls: {len(self.controls)}")
                description_lines.append(f"Union used: {self.use_union}")

            full_description = "\n".join(description_lines)

            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = full_description.encode("utf-8", errors="replace")
            exif_dict["0th"][piexif.ImageIFD.Software] = "UnionSDXL Script - R.D Media".encode("utf-8")

            exif_bytes = piexif.dump(exif_dict)
            self.result.save(output_path, exif=exif_bytes, quality=95, optimize=True)

            self.logger.info("✅ تم حفظ الصورة مع EXIF metadata كامل")

        except Exception as ex:
            self.logger.warning(f"فشل كتابة EXIF metadata: {ex}")

    def run(self):
        try:
            self.logger.info("🚀 بدء تشغيل UnionGenerator...")

            # ────── 1. تحميل الصورة والماسك ────────────────────────────────────
            self.load_image_and_mask()
            self._load_user_controls()

            # 1.1 تطبيق control scales إذا وُجدت
            if getattr(self.args, 'control_scales', None):
                try:
                    scales = [float(x.strip()) for x in self.args.control_scales.split(",")]

                    for i, scale in enumerate(scales):
                        if i < len(self.user_controls):
                            img, typ, _ = self.user_controls[i]
                            self.user_controls[i] = (img, typ, scale)

                    self.logger.info(f"تم تطبيق control scales: {scales}")
                except Exception as e:
                    self.logger.warning(f"فشل تحليل --control-scales: {e}")

            # ────── 2. تحضير الـ controls (مرة واحدة فقط) ────────────────────────
            self.prepare_controls()

            # معلومات عن عدد الـ controls
            if len(self.controls) > 1:
                self.logger.info(f"🚀 تم تفعيل Multi-Control مع {len(self.controls)} controls مختلفة")
            elif len(self.controls) == 1:
                self.logger.info("✅ Control واحد مفعّل")
            else:
                self.logger.warning("⚠️ لا توجد controls صالحة")

            #  التصليح المهم جداً: توحيد أحجام كل الـ controls
            if self.source_img is not None and len(self.controls) > 0:
                target_size = self.get_target_size(self.source_img)
                source_resized = self.safe_resize(self.source_img, target_size)

                resized_controls = []
                for img, typ, scl in self.controls:
                    resized_img = self.resize_control_to_match(img, target_size)
                    resized_controls.append((resized_img, typ, scl))

                self.controls = resized_controls
                self.logger.info(f"✅ تم توحيد أحجام الـ controls إلى {target_size} (حل size mismatch)")

            # ────── 3. حساب use_union بشكل واضح ومنطقي ← هنا المكان الصحيح ──────
            want_union = getattr(self.args, 'use_union', False)
            no_union   = getattr(self.args, 'no_union', False)
            has_controls = len(self.controls) > 0 and not self.controls_disabled

            if no_union:
                self.use_union = False
                self.logger.info("Union معطل صراحة بسبب --no-union")
            elif want_union:
                self.use_union = has_controls
                if not has_controls:
                    self.logger.warning(f"--use-union مفعل لكن لا توجد controls صالحة → Union سيتم تعطيله")
                    if getattr(self.args, 'strict_mode', False):
                        raise ValueError(f"لا يمكن تفعيل Union: {self.controls_disabled_reason or 'لا توجد controls'}")
            else:
                # السلوك الافتراضي: نشغل Union فقط إذا وجدت controls
                self.use_union = has_controls

            self.logger.info(f"قرار نهائي → Union مفعّل: {self.use_union} | Controls: {len(self.controls)} | Disabled: {self.controls_disabled}")

            # ────── 4. التأكد من تحميل الـ pipeline ──────────────────────────────
            self.ensure_pipe()

            # ────── 5. التوليد ────────────────────────────────────────────────────
            self.generate()

            # ────── 6. الحفظ ──────────────────────────────────────────────────────
            self.save()

            self.logger.info("🎉 انتهى التشغيل بنجاح ✓")

        except Exception as e:
            self.logger.critical("💥 فشل كلي في التشغيل", exc_info=True)
            raise

    # ====================== دوال مساعدة عامة (Utilities) ======================
    def ensure_pil_image(self, obj: Any) -> PILImage.Image:
        """تحويل أي مخرج إلى PIL.Image بأمان"""
        if isinstance(obj, PILImage.Image):
            return obj

        if isinstance(obj, np.ndarray):
            return PILImage.fromarray(obj)

        if hasattr(obj, "convert"):
            return obj.convert("RGB")

        if hasattr(obj, "numpy"):
            try:
                arr = obj.numpy()
                if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                return PILImage.fromarray(arr)
            except:
                pass

        self.logger.warning("مخرج غير مدعوم → fallback إلى صورة رمادية")
        return PILImage.new("RGB", (512, 512), (128, 128, 128))

    def get_safe_size(self, img_or_array: Any) -> tuple[int, int]:
        """إرجاع (width, height) بأمان"""
        if hasattr(img_or_array, 'size') and isinstance(img_or_array.size, tuple) and len(img_or_array.size) == 2:
            return img_or_array.size

        if isinstance(img_or_array, np.ndarray) and len(img_or_array.shape) >= 2:
            return img_or_array.shape[1], img_or_array.shape[0]

        if hasattr(img_or_array, 'shape') and len(img_or_array.shape) >= 2:
            return img_or_array.shape[1], img_or_array.shape[0]

        self.logger.warning("حجم غير معروف → استخدام fallback (1024, 1024)")
        return 1024, 1024

    def get_target_size(self, img: PILImage.Image, target_area: int = 1024*1024) -> Tuple[int, int]:
        """يحسب أفضل حجم مع الحفاظ على aspect ratio (مثالي لـ SDXL)"""
        if img is None:
            return 1024, 1024

        w, h = img.size
        aspect = w / h

        # حساب أقرب أبعاد مع الحفاظ على ~1MP
        new_h = int((target_area / aspect) ** 0.5)
        new_w = int(new_h * aspect)

        # جعلها مضاعفات 64 (مطلوب لـ SDXL)
        new_w = (new_w // 64) * 64
        new_h = (new_h // 64) * 64

        if new_w < 512:
            new_w = 512
        if new_h < 512:
            new_h = 512

        return new_w, new_h

    def safe_resize(
        self,
        img: Optional[PILImage.Image],
        size: tuple[int, int],
        resample: PILImage.Resampling = PILImage.Resampling.LANCZOS
    ) -> PILImage.Image:
        """
        تغيير حجم الصورة بأمان مع التعامل مع None
        """
        if img is None:
            self.logger.warning("لا توجد صورة لتغيير حجمها → fallback إلى صورة سوداء")
            return PILImage.new("RGB", size, (0, 0, 0))

        if not isinstance(size, tuple) or len(size) != 2 or not all(isinstance(x, int) for x in size):
            self.logger.warning(f"size غير صالح {size} → استخدام (1024, 1024)")
            size = (1024, 1024)

        try:
            return img.resize(size, resample=resample)
        except Exception as e:
            self.logger.warning(f"فشل تغيير الحجم: {e} → استخدام حجم افتراضي")
            return img.resize((1024, 1024), resample=PILImage.Resampling.LANCZOS)

    def to_pil_image(self, obj: Any) -> PILImage.Image:
        """تحويل أي كائن إلى PIL.Image بأمان"""
        if isinstance(obj, PILImage.Image):
            return obj
        if isinstance(obj, np.ndarray):
            return PILImage.fromarray(obj)
        if hasattr(obj, 'convert'):
            return obj.convert("RGB")
        if hasattr(obj, 'numpy'):
            arr = obj.numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8)
            return PILImage.fromarray(arr)

        self.logger.warning("تحويل غير مدعوم → fallback إلى صورة رمادية")
        return PILImage.new("RGB", (512, 512), (128, 128, 128))

    def safe_save(self, img: PILImage.Image, desired_name: str, extension: str = ".png") -> str:
        """حفظ صورة بأمان مع تجنب تكرار الاسم"""
        from pathlib import Path

        base, ext = os.path.splitext(desired_name)
        if not ext:
            ext = extension

        candidate = Path(desired_name)
        counter = 1
        while candidate.exists():
            candidate = Path(f"{base}_{counter:03d}{ext}")
            counter += 1

        img.save(candidate, quality=95, optimize=True)
        self.logger.info(f"تم الحفظ بنجاح: {candidate}")
        print(f"✓ تم الحفظ في: {candidate}")
        return str(candidate)

    def safe_save_image(self, img: PILImage.Image, base_name: str = "result", extension: str = ".png") -> str:
        """حفظ صورة مع تاريخ وترقيم تلقائي"""
        from pathlib import Path
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        i = 1
        filename = Path(f"{base_name}_{timestamp}_{i:03d}{extension}")

        while filename.exists():
            i += 1
            filename = Path(f"{base_name}_{timestamp}_{i:03d}{extension}")

        img.save(filename, quality=95, optimize=True)
        self.logger.info(f"تم الحفظ: {filename}")
        print(f"✓ تم الحفظ في: {filename}")
        return str(filename)

    def create_simple_mask(self, img: PILImage.Image, threshold: int = 40) -> PILImage.Image:
        """إنشاء قناع بسيط بناءً على الحواف (للتجربة فقط)"""
        if img is None:
            self.logger.warning("لا توجد صورة لإنشاء ماسك → إرجاع ماسك أسود")
            return PILImage.new("L", (1024, 1024), 0)

        try:
            gray = img.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            mask = edges.point(lambda p: 255 if p > threshold else 0)
            self.logger.debug(f"تم إنشاء ماسك بسيط بعتبة {threshold}")
            return mask
        except Exception as e:
            self.logger.error(f"فشل إنشاء الماسك البسيط: {e}")
            return PILImage.new("L", img.size if hasattr(img, 'size') else (1024, 1024), 0)

    def run_with_oom_protection(self, pipe, **kwargs):
        """
        تشغيل الـ inference مع حماية تلقائية من OutOfMemoryError
        يحاول تغيير وضع VRAM عند حدوث OOM
        """
        max_retries = 2
        current_mode = "balanced"

        for attempt in range(max_retries + 1):
            try:
                torch.cuda.empty_cache()
                return pipe(**kwargs)

            except torch.cuda.OutOfMemoryError as oom:
                torch.cuda.empty_cache()

                if attempt == max_retries:
                    self.logger.critical(f"فشل الـ inference بعد {max_retries} محاولات توفير VRAM", exc_info=True)
                    raise RuntimeError("OOM مستمر حتى بعد المحاولات") from oom

                self.logger.warning(f"OOM حدث (محاولة {attempt + 1}/{max_retries}) → تغيير وضع VRAM")

                if current_mode == "balanced":
                    self.logger.info("→ التحويل إلى very_low mode")
                    pipe = self.manager.get_pipeline(
                        low_vram_mode="very_low",
                        force_reload=True
                    )
                    current_mode = "very_low"
                else:
                    self.logger.info("→ التحويل إلى extreme mode (بطيء)")
                    pipe = self.manager.get_pipeline(
                        low_vram_mode="extreme",
                        force_reload=True
                    )
                    current_mode = "extreme"

        return None  # لن يصل هنا عادة

    def update_config(self, dtype: Optional[torch.dtype] = None,
                        device: Optional[Union[str, torch.device]] = None,
                      **kwargs) -> None:
        """
        تحديث الإعدادات الافتراضية للكلاس (dtype, device, إلخ)
        """
        updated = False

        if dtype is not None:
            self.dtype = dtype
            self.logger.info(f"تم تغيير dtype إلى: {dtype}")
            updated = True

        if device is not None:
            self.device = torch.device(device)
            self.logger.info(f"تم تغيير device إلى: {self.device}")
            updated = True

        # يمكن إضافة المزيد من الباراميترات في المستقبل عبر **kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"تم تحديث {key} إلى: {value}")
                updated = True

        if updated:
            # إعادة تحميل الـ pipeline إذا تغير شيء مهم
            if dtype is not None or device is not None:
                self.logger.info("تم تحديث الإعدادات → سيتم إعادة تحميل الـ pipeline عند الحاجة")
                self.pipe = None  # إجبار إعادة التحميل في ensure_pipe
        else:
            self.logger.debug("لم يتم تغيير أي إعدادات")

    def resize_control_to_match(self, control_img: PILImage.Image, target_size: Tuple[int, int]) -> PILImage.Image:
        """تغيير حجم كل control map ليطابق حجم الصورة المصدر المحجمة بالضبط"""
        if control_img.size == target_size:
            return control_img
        self.logger.info(f"Resizing control map from {control_img.size} → {target_size}")
        return control_img.resize(target_size, PILImage.Resampling.LANCZOS)

    def resize_safely(
        self,
        img: PILImage.Image,
        size: tuple[int, int],
        resample: PILImage.Resampling = PILImage.Resampling.LANCZOS
    ) -> PILImage.Image:
        """
        تغيير حجم الصورة بأمان مع تحققات
        """
        if not isinstance(size, tuple) or len(size) != 2 or not all(isinstance(x, int) for x in size):
            self.logger.warning(f"size غير صالح {size} → استخدام (1024, 1024)")
            size = (1024, 1024)

        try:
            return img.resize(size, resample=resample)
        except Exception as e:
            self.logger.warning(f"فشل تغيير الحجم: {e} → استخدام حجم افتراضي")
            return img.resize((1024, 1024), resample=PILImage.Resampling.LANCZOS)

# ────────────────────────────────────────────────
#   غرفة عمليات Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Union ControlNet SDXL Inpainting / Img2Img - R.D Media",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ====================== الباراميترات ======================
    parser.add_argument("--promax", action="store_true",
                        help="استخدام نسخة ControlNet Union ProMax (أقوى)")

    parser.add_argument("--control-start", type=float, default=0.0,
                        help="control_guidance_start (0.0 ~ 1.0)")

    parser.add_argument("--control-end", type=float, default=1.0,
                        help="control_guidance_end (0.0 ~ 1.0)")

    parser.add_argument("--control-scales", type=str, default=None,
                        help='مقاييس الـ controls مفصولة بفاصلة، مثال: "1.0,0.85,0.6"')

    parser.add_argument("--control-openpose", type=str, default=None,
                        help="مسار صورة OpenPose جاهزة")

    parser.add_argument("--control-depth", type=str, default=None,
                        help="مسار صورة Depth جاهزة")

    parser.add_argument("--control-canny", type=str, default=None,
                        help="مسار صورة Canny")

    parser.add_argument("--control-tile", type=str, default=None,
                        help="مسار صورة Tile")

    parser.add_argument("--auto-controls", action="store_true", default=True,
                        help="تفعيل OpenPose + Depth التلقائي (افتراضي)")

    parser.add_argument("--target-area", type=int, default=1024*1024,
                        help="المساحة المستهدفة بالبكسل (مثال: 1048576 = 1024x1024)")

    parser.add_argument("--input", "-i", type=str, default="input.jpg",
                        help="مسار الصورة الأساسية (مطلوب)")

    parser.add_argument("--mask", "-m", type=str, default=None,
                        help="مسار الماسك (اختياري)")

    parser.add_argument("--output-dir", "-o", type=str, default="outputs",
                        help="مجلد حفظ النتائج")

    parser.add_argument("--prompt", type=str,
                        default="masterpiece, best quality, highly detailed, realistic skin, ultra sharp",
                        help="الـ Positive Prompt")

    parser.add_argument("--negative", type=str,
                        default="blurry, low quality, deformed, artifacts, bad anatomy",
                        help="الـ Negative Prompt")

    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed للتكرارية (-1 = عشوائي)")

    parser.add_argument("--strength", type=float, default=0.75,
                        help="قوة التغيير img2img (0.0 - 1.0)")

    parser.add_argument("--steps", type=int, default=30,
                        help="عدد خطوات الـ sampling")

    parser.add_argument("--cfg", type=float, default=7.5,
                        help="Classifier-Free Guidance scale")

    parser.add_argument("--use-union", action="store_true",
                        help="تفعيل ControlNet Union صراحة")

    parser.add_argument("--no-union", action="store_true",
                        help="تعطيل ControlNet Union حتى لو وجدت controls")

    parser.add_argument("--vram-mode", type=str, default="balanced",
                        choices=["off", "balanced", "very_low", "extreme"],
                        help="وضع توفير الـ VRAM")

    parser.add_argument("--compile-unet", action="store_true",
                        help="تفعيل torch.compile للـ UNet (أسرع على RTX 40/50)")

    parser.add_argument("--debug", action="store_true",
                        help="تفعيل وضع التصحيح (DEBUG logging)")

    parser.add_argument("--strict-mode", action="store_true", default=True,
                        help="وضع صارم: يرمي exception بدلاً من التحذيرات")

    args = parser.parse_args()

    # ====================== التنفيذ الرئيسي ======================
    try:
        # إنشاء مولد Union
        generator = UnionGenerator(args)

        # تشغيل العملية كاملة
        generator.run()

        print("\n✅ تم الانتهاء بنجاح!")

    except Exception as e:
        import logging
        logging.error("💥 خطأ أثناء التنفيذ الرئيسي", exc_info=True)
        sys.exit(1)

    finally:
        # تنظيف الذاكرة في النهاية
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # torch.cuda.synchronize()   # اختياري
