# DNA_Repair_Pipeline.py
"""
DNA Repair Pipeline – نظام إصلاح وتحسين الصور بأسلوب DNA-inspired
الطبقة الأولى: Input Prompt + Filter + Scene DNA Genes
"""

# ────────────────────────────────────────────────
#          Requirements (انسخها في requirements.txt)
# ────────────────────────────────────────────────
"""
torch>=2.0.0
torchvision
torchaudio
diffusers>=0.20.0
transformers>=4.35.0
accelerate>=0.25.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
torchmetrics>=0.11.0
# اختياري إذا بتستخدم CLIP أو vision models بشكل كبير:
# timm
# ftfy
# regex
"""

import re
import sys
import json
import random
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Literal, Any

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageResampling

import torch
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from transformers import CLIPProcessor, CLIPModel

# ── Globals for availability checks ──────────────────────────────────
DIFFUSERS_AVAILABLE = True
CLIP_AVAILABLE = True

try:
    # Already imported
    pass
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("تحذير: diffusers غير مثبتة → الـ diffusion محدود")

try:
    # Already imported
    pass
except ImportError:
    CLIP_AVAILABLE = False
    print("تحذير: transformers غير مثبتة → الـ genes extraction محدود")

class DNARepairPipeline:
    """
    الكلاس الرئيسي للـ DNA Repair Pipeline.
    يحمل النماذج مرة واحدة، يدير الـ state، ويحتوي على methods لكل مرحلة.
    """

    def __init__(
        self,
        controlnet_repo: str = "lllyasviel/sd-controlnet-union-sdxl-1.0",
        inpaint_repo: str = "runwayml/stable-diffusion-inpainting",
        clip_repo: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        dtype = torch.float16,
        lowvram: bool = False,
        load_on_init: bool = True,           # جديد: هل نحمل النماذج فورًا أم نؤجل
    ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.lowvram = lowvram

        # ── State ────────────────────────────────────────────────────────
        self.current_image: Optional[Image.Image] = None
        self.current_mask: Optional[Image.Image] = None
        self.current_prompt: str = ""
        self.dna_genes: Dict[str, str] = {}
        self.monitoring_report: Dict[str, Any] = {}

        # ── Models ───────────────────────────────────────────────────────
        self.controlnet: Optional[ControlNetModel] = None
        self.pipeline: Optional[StableDiffusionControlNetInpaintPipeline] = None
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None

        self._models_loaded = False

        if load_on_init:
            self._load_models(
                controlnet_repo=controlnet_repo,
                inpaint_repo=inpaint_repo,
                clip_repo=clip_repo
            )

    def _load_models(
        self,
        controlnet_repo: str,
        inpaint_repo: str,
        clip_repo: str,
    ) -> bool:
        """
        Returns: True إذا نجح التحميل الكامل أو الجزئي، False إذا فشل كل شيء
        """
        if self._models_loaded:
            print("النماذج محملة بالفعل.")
            return True

        success = True
        loaded_components = []

        # 1. ControlNet
        if DIFFUSERS_AVAILABLE:
            try:
                print(f"جاري تحميل ControlNet: {controlnet_repo}")
                self.controlnet = ControlNetModel.from_pretrained(
                    controlnet_repo,
                    torch_dtype=self.dtype,
                    use_safetensors=True,  # أكثر أمانًا وسرعة غالبًا
                ).to(self.device)
                loaded_components.append("ControlNet")
            except Exception as e:
                print(f"فشل تحميل ControlNet: {e}")
                success = False

        # 2. Inpaint Pipeline
        if DIFFUSERS_AVAILABLE and self.controlnet is not None:
            try:
                print(f"جاري تحميل Inpaint Pipeline: {inpaint_repo}")
                self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    inpaint_repo,
                    controlnet=self.controlnet,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    use_safetensors=True,
                ).to(self.device)

                self.pipeline.enable_attention_slicing()

                if self.device.startswith("cuda"):
                    if self.lowvram:
                        print("وضع low VRAM مفعّل → sequential CPU offload")
                        self.pipeline.enable_sequential_cpu_offload()
                    else:
                        print("تفعيل model CPU offload")
                        self.pipeline.enable_model_cpu_offload()

                loaded_components.append("Inpaint Pipeline")
            except Exception as e:
                print(f"فشل تحميل Pipeline: {e}")
                success = False
        elif DIFFUSERS_AVAILABLE:
            print("لم يتم تحميل ControlNet → Pipeline لن يُحمّل")

        # 3. CLIP
        if CLIP_AVAILABLE:
            try:
                print(f"جاري تحميل CLIP: {clip_repo}")
                self.clip_model = CLIPModel.from_pretrained(clip_repo).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(clip_repo)
                loaded_components.append("CLIP")
            except Exception as e:
                print(f"فشل تحميل CLIP: {e}")
                success = False  # ليس حرجًا مثل الـ diffusion، لكن نعلم المستخدم

        # تقرير التحميل
        if loaded_components:
            print("تم تحميل: " + ", ".join(loaded_components))
        else:
            print("لم يتم تحميل أي نموذج رئيسي.")

        self._models_loaded = bool(loaded_components)
        return self._models_loaded

    def ensure_models_loaded(self) -> bool:
        """
        تستدعى قبل أي عملية تحتاج النماذج
        تسمح بتحميل متأخر إذا تم تعطيل التحميل في __init__
        """
        if not self._models_loaded:
            print("النماذج غير محملة بعد → جاري التحميل الآن...")
            # هنا يمكنك تمرير القيم الافتراضية أو جعلها تأخذ من الـ init
            return self._load_models(
                controlnet_repo="lllyasviel/sd-controlnet-union-sdxl-1.0",
                inpaint_repo="runwayml/stable-diffusion-inpainting",
                clip_repo="openai/clip-vit-base-patch32",
            )
        return True

    def generate_controlnet_net(
        self,
        input_image: Image.Image,
        mask: Image.Image,
        prompt: str = "detailed structural grid, clean edges, high contrast",
        negative_prompt: str = "blurry, noisy, deformed, artifacts",
        net_strength: float = 0.68,
        steps: int = 18,
        guidance_scale: float = 7.2,
    ) -> Image.Image:
        """
        إنتاج خريطة هيكلية (Net) باستخدام ControlNet داخل حدود الماسك
        """
        if not self.pipeline:
            return self._quick_sharpen_fallback(input_image, mask)

        img, msk = self._prepare_image_and_mask(input_image, mask)
        if msk is None:
            return img

        # إذا كان lowvram مفعل → نعدل القيم مسبقاً
        effective_steps = max(10, steps // 2) if self.lowvram else steps
        effective_strength = net_strength * 0.85 if self.lowvram else net_strength
        effective_guidance = guidance_scale * 0.9 if self.lowvram else guidance_scale

        result = None

        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=msk,
                control_image=img,           # union ← الصورة نفسها
                strength=effective_strength,
                num_inference_steps=effective_steps,
                guidance_scale=effective_guidance,
            ).images[0]

        except torch.cuda.OutOfMemoryError:
            if self.device.startswith("cuda"):
                try:
                    self.pipeline.to("cpu")
                    self.device = "cpu"
                    self.pipeline.enable_model_cpu_offload()

                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=img,
                        mask_image=msk,
                        control_image=img,
                        strength=effective_strength * 0.8,
                        num_inference_steps=max(8, effective_steps // 2),
                        guidance_scale=effective_guidance * 0.85,
                    ).images[0]
                except:
                    pass  # لو فشل حتى كده → نروح للـ fallback

        except Exception:
            pass  # أي خطأ آخر → fallback

        if result is None:
            return self._net_fallback(img, msk)

        return ImageEnhance.Sharpness(result).enhance(1.10)


    def geometric_repair_via_net(
        self,
        input_image: Image.Image,
        mask: Image.Image,
        net: Image.Image,
        prompt: str = "perfect anatomy, sharp edges, correct proportions",
        negative_prompt: str = "deformed, bad anatomy, blurry, artifacts",
        strength: float = 0.32,
        steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        إعادة بناء الشكل الهندسي بدقة عالية باستخدام Net كـ control
        """
        if not self.pipeline:
            return self._quick_sharpen_fallback(input_image, mask)

        img, msk = self._prepare_image_and_mask(input_image, mask)
        if msk is None:
            return img

        net = net.convert("RGB").resize(img.size, Image.Resampling.LANCZOS)

        # تعديل مسبق لـ lowvram
        effective_steps = max(10, steps // 2) if self.lowvram else steps
        effective_strength = strength * 0.75 if self.lowvram else strength
        effective_guidance = guidance_scale * 0.9 if self.lowvram else guidance_scale

        result = None

        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=msk,
                control_image=net,
                strength=effective_strength,
                num_inference_steps=effective_steps,
                guidance_scale=effective_guidance,
            ).images[0]

        except torch.cuda.OutOfMemoryError:
            if self.device.startswith("cuda"):
                try:
                    self.pipeline.to("cpu")
                    self.device = "cpu"
                    self.pipeline.enable_model_cpu_offload()

                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=img,
                        mask_image=msk,
                        control_image=net,
                        strength=effective_strength * 0.75,
                        num_inference_steps=max(8, effective_steps // 2),
                        guidance_scale=effective_guidance * 0.85,
                    ).images[0]
                except:
                    pass

        except Exception:
            pass

        if result is None:
            return self._repair_fallback(img, msk)

        return ImageEnhance.Contrast(result).enhance(1.04)


    def _quick_sharpen_fallback(self, img: Image.Image, mask: Optional[Image.Image]) -> Image.Image:
        enhanced = ImageEnhance.Sharpness(img).enhance(1.20)
        if mask is None:
            return enhanced
        return Image.composite(enhanced, img, mask)


    def _net_fallback(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        """fallback مخصص لـ generate_controlnet_net (تركيز على الحواف)"""
        from cv2 import cvtColor, Canny, dilate
        arr = np.array(img)
        gray = cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = Canny(gray, 70, 180)
        edges = dilate(edges, None, iterations=1)
        edges_img = Image.fromarray(edges).convert("RGB")
        sharp = ImageEnhance.Sharpness(edges_img).enhance(1.5)
        return Image.composite(sharp, img, mask)


    def _repair_fallback(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        """fallback مخصص لـ geometric_repair_via_net (تركيز على التباين والتفاصيل)"""
        contrast = ImageEnhance.Contrast(img).enhance(1.30)
        sharp = ImageEnhance.Sharpness(contrast).enhance(1.25)
        return Image.composite(sharp, img, mask)


    def _cpu_fallback_net(self, image, mask, prompt):
        """بديل خفيف جدًا بدون diffusion"""
        image, mask = self._prepare_image_and_mask(image, mask)

        if mask is None:
            return image.filter(ImageFilter.SHARPEN)

        # مثال بسيط: edge detection + contrast + sharpen داخل الماسك
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Canny edges
        edges = cv2.Canny(gray, 80, 200)
        edges = cv2.dilate(edges, None, iterations=1)

        # تحويل لـ PIL
        edges_pil = Image.fromarray(edges).convert("RGB")

        # تلميع
        enhancer = ImageEnhance.Contrast(edges_pil)
        enhanced = enhancer.enhance(1.4)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)

        # دمج مع الماسك
        result = Image.composite(enhanced, image, mask)


    def get_user_input() -> Tuple[Image.Image, str, Optional[Image.Image]]:
        """
        جمع المدخلات الأساسية من المستخدم أو من وسيط سطر الأوامر

        Returns:
            (input_image, raw_prompt, mask_image or None)
        """
        # ── طريقة 1: تشغيل تفاعلي (للتجربة السريعة) ────────────────────────
        if len(sys.argv) <= 1:
            print("=== DNA Repair Pipeline – مرحبًا ===")
            print("أدخل مسار الصورة الأصلية (أو اتركه فارغًا للتجربة):")
            img_path = input("> ").strip()

            if not img_path:
                print("لم يتم إدخال صورة → سيتم استخدام صورة اختبار وهمية")
                input_image = Image.new("RGB", (512, 512), color=(120, 140, 180))
            else:
                try:
                    input_image = Image.open(img_path).convert("RGB")
                    print(f"تم تحميل الصورة: {img_path}")
                except Exception as e:
                    print(f"خطأ في تحميل الصورة: {e}")
                    sys.exit(1)

            print("\nأدخل الـ Prompt (الوصف):")
            raw_prompt = input("> ").strip()
            if not raw_prompt:
                raw_prompt = "وجه واقعي جميل لفتاة في غابة سحرية، إضاءة ذهبية ناعمة، تفاصيل عالية"

            print("\nهل لديك ماسك (مسار الملف)؟ (اضغط Enter إذا لا)")
            mask_path = input("> ").strip()
            mask = None
            if mask_path:
                try:
                    mask = Image.open(mask_path).convert("L")
                    print(f"تم تحميل الماسك: {mask_path}")
                except Exception as e:
                    print(f"خطأ في تحميل الماسك: {e} → سيتم الاستمرار بدون ماسك")

            return input_image, raw_prompt, mask

        # ── طريقة 2: تشغيل من سطر الأوامر (للسكريبتات والأتمتة) ──────────────
        else:
            import argparse

            parser = argparse.ArgumentParser(description="DNA Repair Pipeline – المدخل الأساسي")
            parser.add_argument("--image", required=True, help="مسار الصورة الأصلية")
            parser.add_argument("--prompt", required=True, help="الوصف النصي (prompt)")
            parser.add_argument("--mask", help="مسار الماسك (اختياري)")
            args = parser.parse_args()

            try:
                input_image = Image.open(args.image).convert("RGB")
            except Exception as e:
                print(f"خطأ في تحميل الصورة: {e}")
                sys.exit(1)

            raw_prompt = args.prompt

            mask = None
            if args.mask:
                try:
                    mask = Image.open(args.mask).convert("L")
                except Exception as e:
                    print(f"خطأ في تحميل الماسك: {e} → سيتم الاستمرار بدون ماسك")

            return input_image, raw_prompt, mask


    def filter_and_enhance_prompt(
        raw_prompt: str,
        style_preferences: Optional[List[str]] = None,
        add_quality_boosters: bool = True,
        remove_negatives: bool = True,
    ) -> str:
        """
        تنقية وتحسين الـ prompt الخام قبل أي خطوة لاحقة

        Args:
            raw_prompt: النص الذي كتبه المستخدم
            style_preferences: قائمة أساليب إضافية (مثل: ["cinematic", "oil painting"])
            add_quality_boosters: هل نضيف كلمات مثل masterpiece, 8k, highly detailed؟
            remove_negatives: هل نزيل كلمات سلبية شائعة (blurry, deformed, ...)؟

        Returns:
            prompt منظف ومحسن
        """
        if not raw_prompt:
            return "وجه واقعي جميل، تفاصيل عالية، إضاءة طبيعية، ألوان حيوية"

        prompt = raw_prompt.strip()

        # 1. إزالة كلمات سلبية شائعة (اختياري)
        if remove_negatives:
            bad_words = [
                "blurry", "low quality", "ugly", "deformed", "bad anatomy",
                "extra limbs", "lowres", "worst quality", "poorly drawn"
            ]
            for word in bad_words:
                prompt = prompt.replace(word, "").replace(word.capitalize(), "")

        # 2. إضافة معززات جودة عامة (اختياري)
        if add_quality_boosters:
            boosters = [
                "masterpiece", "best quality", "highly detailed",
                "sharp focus", "8k", "ultra detailed", "cinematic lighting",
                "vibrant colors", "realistic textures"
            ]
            prompt += ", " + ", ".join(boosters)

        # 3. إضافة تفضيلات أسلوبية إذا وُجدت
        if style_preferences and isinstance(style_preferences, list):
            prompt += ", " + ", ".join(style_preferences)

        # 4. تنظيف نهائي (مسافات، فواصل زائدة)
        prompt = " ".join(prompt.split())
        prompt = prompt.replace(" ,", ",").replace(",,", ",").strip()

        return prompt


    def extract_initial_dna_genes(
        self,
        filtered_prompt: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, str]:
        """
        استخراج جينات المشهد الأولية باستخدام CLIP zero-shot classification
        (fallback إلى keyword matching إذا لم يكن CLIP متوفرًا)
        """
        genes: Dict[str, str] = {}

        text = filtered_prompt.strip()

        if self.clip_model and self.clip_processor:
            # ── استخدام CLIP لتصنيف أكثر دقة ────────────────────────────────
            def classify(candidates: List[str]) -> str:
                if not candidates:
                    return "unknown"
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                text_features = self.clip_model.get_text_features(**inputs)
                label_inputs = self.clip_processor(text=candidates, return_tensors="pt", padding=True).to(self.device)
                label_features = self.clip_model.get_text_features(**label_inputs)
                sim = (text_features @ label_features.T).softmax(dim=-1)[0]
                return candidates[sim.argmax().item()]

            # 1. Pose
            pose_cand = ["standing pose", "sitting pose", "lying pose", "kneeling pose", "crouching pose", "neutral pose"]
            genes["pose_gene"] = classify(pose_cand)

            # 2. Mood
            mood_cand = ["mysterious mood", "serene mood", "calm mood", "epic mood", "romantic mood", "sad mood", "joyful mood", "neutral mood"]
            genes["mood_gene"] = classify(mood_cand)

            # 3. Lighting
            light_cand = ["golden hour lighting", "dramatic lighting", "soft lighting", "cinematic lighting", "natural lighting", "neon lighting", "neutral lighting"]
            genes["lighting_gene"] = classify(light_cand)

            # 4. Color
            color_cand = ["warm colors", "cool colors", "pastel colors", "vibrant colors", "muted colors", "monochrome", "natural colors"]
            genes["color_gene"] = classify(color_cand)

            # 5. Style
            style_cand = ["realistic style", "photorealistic style", "cinematic style", "painting style", "anime style", "digital art style", "fantasy style"]
            genes["style_gene"] = classify(style_cand)

            # 6. Environment
            env_cand = ["forest environment", "beach environment", "mountain environment", "city environment", "desert environment", "indoor environment", "neutral environment"]
            genes["environment_gene"] = classify(env_cand)

        else:
            # ── Fallback بسيط (الكود القديم المحسن قليلاً) ─────────────────────
            text_lower = text.lower()

            # pose
            if any(k in text_lower for k in ["standing", "upright"]): genes["pose_gene"] = "standing"
            elif any(k in text_lower for k in ["sitting", "seated"]): genes["pose_gene"] = "sitting"
            elif any(k in text_lower for k in ["lying", "reclining"]): genes["pose_gene"] = "lying"
            else: genes["pose_gene"] = "neutral pose"

            # mood (أول تطابق فقط)
            mood_keywords = ["mysterious", "serene", "calm", "epic", "romantic", "sad", "joyful", "dark"]
            genes["mood_gene"] = next((m for m in mood_keywords if m in text_lower), "neutral")

            # lighting, color, style, env ... نفس الفكرة (يمكن نسخ/لصق مع تعديل بسيط)

        # placeholder لتحليل الصورة (مستقبلي)
        if image is not None:
            genes["image_analyzed"] = "pending CLIP/VLM integration"

        return genes


    def prepare_prompt_and_genes(
        self,
        raw_prompt: str,
        image: Optional[Image.Image] = None,
        style_preferences: Optional[List[str]] = None,
        add_quality_boosters: bool = True,
        remove_negatives: bool = True,
    ) -> Tuple[str, Dict[str, str]]:
        """
        الطبقة الأولى: تنقية + استخراج جينات
        """
        filtered = self.filter_and_enhance_prompt(
            raw_prompt=raw_prompt,
            style_preferences=style_preferences,
            add_quality_boosters=add_quality_boosters,
            remove_negatives=remove_negatives,
        )

        genes = self.extract_initial_dna_genes(
            filtered_prompt=filtered,
            image=image
        )

        # تحديث الـ state إذا كنت تستخدمه
        self.current_prompt = filtered
        self.dna_genes = genes

        return filtered, genes


    def _clip_classify(self, text: str, candidates: list[str]) -> str:
        """مساعد داخلي لتصنيف النص باستخدام CLIP zero-shot"""
        inputs = self.clip_processor(text=[text], images=None, return_tensors="pt", padding=True).to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)

        # تصنيف مقابل الـ candidates
        label_inputs = self.clip_processor(text=candidates, return_tensors="pt", padding=True).to(self.device)
        label_features = self.clip_model.get_text_features(**label_inputs)

        # حساب التشابه
        similarities = (text_features @ label_features.T).softmax(dim=-1).cpu().detach().numpy()[0]
        best_idx = similarities.argmax()
        return candidates[best_idx]


    def load_controlnet_pipeline(
        controlnet_repo: str = "lllyasviel/sd-controlnet-union-sdxl-1.0",
        inpaint_repo: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    ) -> tuple[ControlNetModel, StableDiffusionControlNetInpaintPipeline]:
        """
        تحميل ControlNet + Inpaint Pipeline مرة واحدة (يفضل تخزينه خارج الدالة إذا أمكن)
        """
        print(f"تحميل ControlNet من: {controlnet_repo}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_repo,
            torch_dtype=dtype,
        ).to(device)

        print(f"تحميل Inpaint Pipeline من: {inpaint_repo}")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            inpaint_repo,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

        return controlnet, pipe


    def generate_controlnet_net(
        self,
        input_image: Image.Image,
        mask: Image.Image,
        prompt: str = "detailed structural grid, clean edges, high contrast",
        negative_prompt: str = "blurry, noisy, low detail, deformed",
        control_type: Literal["union", "canny", "lineart", "depth", "tile"] = "union",
        net_strength: float = 0.68,
        steps: int = 18,
        guidance_scale: float = 7.2,
    ) -> Image.Image:
        """
        إنتاج خريطة هيكلية (Net) باستخدام ControlNet داخل حدود الماسك
        """
        if not self.pipeline:
            return self._edge_fallback(input_image, mask, control_type)

        img, msk = self._prepare_image_and_mask(input_image, mask)
        if msk is None:
            return img

        # تعديل مسبق لـ lowvram أو cpu
        eff_steps = max(10, steps // 2) if self.lowvram or self.device == "cpu" else steps
        eff_strength = net_strength * 0.85 if self.lowvram else net_strength

        control_img = img  # default (union)

        if control_type == "canny":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 180)
            control_img = Image.fromarray(edges).convert("RGB")
        elif control_type == "lineart":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            lineart = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            control_img = Image.fromarray(lineart).convert("RGB")
        elif control_type in ["depth", "tile"]:
            control_img = img  # placeholder / fallback

        try:
            out = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=msk,
                control_image=control_img,
                strength=eff_strength,
                num_inference_steps=eff_steps,
                guidance_scale=guidance_scale,
            ).images[0]

            return ImageEnhance.Sharpness(out).enhance(1.12)

        except torch.cuda.OutOfMemoryError:
            if self.device.startswith("cuda"):
                try:
                    self.pipeline.to("cpu")
                    self.device = "cpu"
                    self.pipeline.enable_model_cpu_offload()
                    out = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=img,
                        mask_image=msk,
                        control_image=control_img,
                        strength=eff_strength * 0.75,
                        num_inference_steps=max(8, eff_steps // 2),
                        guidance_scale=guidance_scale * 0.9,
                    ).images[0]
                    return ImageEnhance.Sharpness(out).enhance(1.12)
                except:
                    pass
            return self._edge_fallback(img, msk, control_type)

        except Exception:
            return self._edge_fallback(img, msk, control_type)


    def _edge_fallback(self, img: Image.Image, mask: Image.Image, control_type: str) -> Image.Image:
        """fallback خفيف وسريع حسب نوع الـ control"""
        if control_type in ["canny", "lineart"]:
            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if control_type == "canny":
                edges = cv2.Canny(gray, 70, 180)
            else:
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            edges_img = Image.fromarray(edges).convert("RGB")
            sharp = ImageEnhance.Sharpness(edges_img).enhance(1.4)
        else:
            sharp = ImageEnhance.Sharpness(img).enhance(1.25)

        return Image.composite(sharp, img, mask) if mask else sharp


    def create_dna_light_layer(
        self,
        size: Tuple[int, int],
        base_color: Tuple[int, int, int] = (80, 220, 120),
        opacity: float = 0.35,
        gradient_direction: Literal["radial", "horizontal", "vertical", "none"] = "radial",
        add_grain: bool = True,
        grain_intensity: float = 0.03,
    ) -> Image.Image:
        """
        إنشاء طبقة DNA Light شفافة (radial gradient + grain اختياري)
        """
        w, h = size
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        overlay = Image.new("RGB", size, base_color)
        draw = ImageDraw.Draw(overlay)

        if gradient_direction == "radial":
            cx, cy = w // 2, h // 2
            max_r = max(w, h) / 2
            for y in range(h):
                for x in range(w):
                    d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                    r = min(1.0, d / max_r)
                    alpha = int(255 * (1 - r ** 1.5))
                    col = tuple(int(c * (1 - r * 0.4)) for c in base_color)
                    draw.point((x, y), fill=col + (alpha,))
        elif gradient_direction == "horizontal":
            for x in range(w):
                r = x / w
                alpha = int(255 * (1 - abs(r - 0.5) * 2))
                col = tuple(int(c * (1 - abs(r - 0.5) * 0.6)) for c in base_color)
                draw.line((x, 0, x, h), fill=col + (alpha,))
        elif gradient_direction == "vertical":
            for y in range(h):
                r = y / h
                alpha = int(255 * (1 - abs(r - 0.5) * 2))
                col = tuple(int(c * (1 - abs(r - 0.5) * 0.6)) for c in base_color)
                draw.line((0, y, w, y), fill=col + (alpha,))
        # none → ما بنعملش تدرج، نستخدم اللون الثابت

        if add_grain:
            grain = np.random.normal(0, grain_intensity * 255, (h, w, 3)).astype(np.int16)
            ov_arr = np.array(overlay, dtype=np.int16)
            ov_arr[..., :3] = np.clip(ov_arr[..., :3] + grain, 0, 255)
            overlay = Image.fromarray(ov_arr.astype(np.uint8))

        overlay.putalpha(int(255 * opacity))
        layer.paste(overlay, (0, 0), overlay)

        return layer


    def geometric_repair_via_net(
        self,
        input_image: Image.Image,
        mask: Image.Image,
        net: Image.Image,
        prompt: str = (
            "detailed realistic anatomy, perfect proportions, sharp edges, "
            "correct perspective, high structural accuracy, clean geometry"
        ),
        negative_prompt: str = (
            "deformed, bad anatomy, extra limbs, warped, blurry, low detail, "
            "distorted face, asymmetry, artifacts"
        ),
        strength: float = 0.32,
        steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        إعادة بناء هندسي دقيق باستخدام Net كـ control.
        يعتمد على النماذج المحملة مسبقًا في الكلاس.
        """
        if not self.pipeline:
            print("Pipeline غير متوفر → fallback بسيط")
            return self._repair_fallback(input_image, mask)

        # تهيئة الصور + ماسك + net مرة واحدة
        img, msk = self._prepare_image_and_mask(input_image, mask)
        if msk is None:
            return img

        net = net.convert("RGB").resize(img.size, Image.Resampling.LANCZOS)

        # تعديل تلقائي لو موارد ضعيفة
        eff_steps    = max(10, steps // 2) if self.lowvram or self.device == "cpu" else steps
        eff_strength = strength * 0.8 if self.lowvram else strength
        eff_guidance = guidance_scale * 0.9 if self.lowvram else guidance_scale

        result = None

        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=msk,
                control_image=net,
                strength=eff_strength,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
            ).images[0]

        except torch.cuda.OutOfMemoryError:
            print("OOM → محاولة خفيفة على CPU")
            try:
                if self.device.startswith("cuda"):
                    self.pipeline.to("cpu")
                    self.device = "cpu"
                    self.pipeline.enable_model_cpu_offload()

                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=img,
                    mask_image=msk,
                    control_image=net,
                    strength=eff_strength * 0.75,
                    num_inference_steps=max(8, eff_steps // 2),
                    guidance_scale=eff_guidance * 0.85,
                ).images[0]
            except:
                pass  # لو فشل → fallback

        except Exception as e:
            print(f"خطأ في الإصلاح الهندسي: {type(e).__name__}")
            # يمكن تسجيل الخطأ في log لو عندك

        if result is None:
            return self._repair_fallback(img, msk)

        # تلميع خفيف وفعّال
        return ImageEnhance.Contrast(
            ImageEnhance.Sharpness(result).enhance(1.08)
        ).enhance(1.04)


    def _repair_fallback(self, img: Image.Image, mask: Optional[Image.Image]) -> Image.Image:
        """
        fallback بسيط وسريع للإصلاح الهندسي (تباين + حدة داخل الماسك)
        """
        enhanced = ImageEnhance.Contrast(img).enhance(1.25)
        sharp    = ImageEnhance.Sharpness(enhanced).enhance(1.20)

        if mask is None:
            return sharp

        return Image.composite(sharp, img, mask)


    def add_colored_dna_layers(
        self,
        net_image: Image.Image,
        mask: Image.Image,
        base_colors: List[Tuple[int, int, int]] = [
            (220, 60, 60),   # أحمر – طاقة
            (60, 220, 60),   # أخضر – نمو
            (60, 60, 220)    # أزرق – توازن
        ],
        blend_mode: Literal["density", "wave", "helix"] = "density",
        opacity_base: float = 0.52,
        edge_thickness: int = 4,
        edge_highlight_color: Tuple[int, int, int] = (220, 240, 255),
    ) -> Image.Image:
        """
        تلوين أضلاع الشبكة بألوان DNA-inspired مع شفافية وإبراز.
        """
        w, h = net_image.size
        net_arr = np.array(net_image.convert("RGB"))

        # استخراج الحواف (edges)
        gray = cv2.cvtColor(net_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 140)
        if edge_thickness > 1:
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # تحضير الماسك كـ float [0,1]
        if mask is None:
            mask_arr = np.ones((h, w), dtype=np.float32)
        else:
            mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

        # الطبقة الملونة (RGBA)
        colored = np.zeros((h, w, 4), dtype=np.uint8)

        if blend_mode == "density":
            intensity = mask_arr ** 1.4
            for i, color in enumerate(base_colors):
                wgt = intensity * (0.75 - i * 0.25)
                colored[..., :3] += (np.array(color) * wgt[..., None]).astype(np.uint8)

        elif blend_mode in ["wave", "helix"]:
            x = np.linspace(0, 40 * np.pi, w)
            y = np.linspace(0, 20 * np.pi, h)[:, None]
            wave = (np.sin(x) + np.sin(y) + 2) / 4
            wave = np.repeat(wave[..., None], 3, axis=2)

            mixed = np.zeros((h, w, 3))
            for color in base_colors:
                mixed += np.array(color) * wave * (1 / len(base_colors))

            colored[..., :3] = (mixed * mask_arr[..., None]).astype(np.uint8)

        # الشفافية الأساسية
        colored[..., 3] = (mask_arr * 255 * opacity_base).astype(np.uint8)

        # إبراز الحواف
        colored[edges > 0, :3] = edge_highlight_color
        colored[edges > 0, 3] = 240

        return Image.fromarray(colored)


    def dna_inspired_color_pulse(
        self,
        image: Image.Image,
        mask: Image.Image,
        pulse_steps: int = 6,
        positive_boost: float = 0.28,
        negative_suppress: float = 0.22,
        pulse_factor_decay: float = 0.65,
        hue_mutation_strength: float = 8.0,
        use_latent: bool = False,
    ) -> Image.Image:
        """
        نبض DNA-inspired لتحسين الألوان والحيوية داخل الماسك فقط (pixel-level).
        """
        if use_latent:
            raise NotImplementedError("latent pulse غير مدعوم حاليًا")

        # تحضير أولي مرة واحدة
        img = image.convert("RGB")
        msk = mask.convert("L") if mask is not None else Image.new("L", img.size, 255)

        if img.size != msk.size:
            msk = msk.resize(img.size, Image.Resampling.LANCZOS)

        arr = np.array(img, dtype=np.float32)
        mask_arr = np.array(msk, dtype=np.float32) / 255.0
        mask_3d = mask_arr[..., np.newaxis]  # مرة واحدة فقط

        # تحويل مرة واحدة إلى HSV
        hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        for step in range(pulse_steps):
            factor = max(0.05, 1.0 - (step / pulse_steps) * pulse_factor_decay)

            # hue shift (عشوائي خفيف)
            hue_shift = np.random.normal(0, hue_mutation_strength * factor, size=mask_arr.shape)
            hsv[..., 0] = (hsv[..., 0] + hue_shift * mask_arr) % 180

            # boosts وقمع (vectorized)
            sat_boost = 1.0 + positive_boost * factor * mask_arr
            val_boost = 1.0 + positive_boost * 0.18 * factor * mask_arr
            suppress   = 1.0 - negative_suppress * factor * mask_arr

            hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost * suppress, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * val_boost, 0, 255)

        # عودة لـ RGB + clip نهائي
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)


    def some_function(self, input_image: Image.Image, mask: Image.Image, ...):
        if mask is None:
            # fallback: نعمل ماسك كامل أو نلغي العملية
            print("تحذير: لا يوجد ماسك → سيتم تجاهل الـ inpainting")
            return input_image  # أو نرجع صورة معدلة بطريقة أخرى

        # التأكد من الحجم والـ mode
        try:
            input_image = input_image.convert("RGB")
            mask = mask.convert("L")

            if input_image.size != mask.size:
                print(f"تعديل حجم الماسك من {mask.size} إلى {input_image.size}")
                mask = mask.resize(input_image.size, Image.Resampling.LANCZOS)

        except Exception as e:
            print(f"خطأ في تهيئة الصورة/الماسك: {e}")
            return input_image  # fallback مهم جدًا هنا


    def monitor_repair_result(
        original: Image.Image,
        repaired: Image.Image,
        mask: Optional[Image.Image] = None,
        max_lpips_threshold: float = 0.30,
        min_ssim_threshold: float = 0.92,
        detail_loss_threshold: float = 0.08,
        color_dev_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """
        تقييم النتيجة بعد الإصلاح باستخدام عدة مقاييس (SSIM + LPIPS + detail + color)
        """
        original = original.convert("RGB")
        repaired = repaired.convert("RGB")

        orig_np = np.array(original)
        rep_np = np.array(repaired)

        report: Dict[str, Any] = {
            "ssim_score": None,
            "lpips_score": None,
            "detail_loss_ratio": None,
            "color_deviation": None,
            "overall_quality_score": None,
            "needs_correction": False,
            "issues": []
        }

        # ── 1. SSIM ────────────────────────────────────────────────────────
        try:
            orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
            rep_gray  = cv2.cvtColor(rep_np,  cv2.COLOR_RGB2GRAY)

            report["ssim_score"] = float(ssim(
                orig_gray, rep_gray,
                data_range=orig_gray.max() - orig_gray.min(),
                full=False
            ))
        except Exception as e:
            print(f"SSIM فشل: {e}")
            report["ssim_score"] = 0.0

        # ── 2. LPIPS (الآن مفعّل فعلياً) ──────────────────────────────────
        try:
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type='alex',  # أو 'vgg' أو 'squeeze'
                reduction='mean'
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # تحويل إلى tensor [0,1] و [B,C,H,W]
            orig_tensor = torch.from_numpy(orig_np).permute(2,0,1).unsqueeze(0).float() / 255.0
            rep_tensor  = torch.from_numpy(rep_np).permute(2,0,1).unsqueeze(0).float() / 255.0

            if torch.cuda.is_available():
                orig_tensor = orig_tensor.cuda()
                rep_tensor  = rep_tensor.cuda()

            lpips_val = lpips_metric(orig_tensor, rep_tensor).item()
            report["lpips_score"] = float(lpips_val)
        except Exception as e:
            print(f"LPIPS فشل: {e}")
            report["lpips_score"] = 0.35   # قيمة افتراضية سيئة نسبياً

        # ── 3. Detail loss (Laplacian variance) ─────────────────────────────
        lap_orig = float(cv2.Laplacian(orig_gray, cv2.CV_64F).var())
        lap_rep  = float(cv2.Laplacian(rep_gray,  cv2.CV_64F).var())

        if lap_orig > 1e-6:
            detail_loss = max(0.0, (lap_orig - lap_rep) / lap_orig)
        else:
            detail_loss = 0.0

        report["detail_loss_ratio"] = float(detail_loss)

        # ── 4. Color deviation (Wasserstein distance على الهيستوغرام) ─────
        color_dev = 0.0
        for ch in range(3):
            h1 = cv2.calcHist([orig_np], [ch], None, [256], [0, 256]).flatten()
            h2 = cv2.calcHist([rep_np],  [ch], None, [256], [0, 256]).flatten()
            color_dev += float(wasserstein_distance(h1, h2))
        report["color_deviation"] = float(color_dev / 3.0)

        # ── 5. حساب درجة الجودة الكلية (أكثر توازناً) ─────────────────────
        # نريد قيمة بين 0 و 100 تقريباً
        # كلما كانت أعلى → أفضل

        ssim_norm   = report["ssim_score"]               # 0–1
        lpips_norm  = 1.0 - min(report["lpips_score"], 1.0)   # عكس LPIPS (أقل أفضل)
        detail_norm = 1.0 - min(report["detail_loss_ratio"], 1.0)
        color_norm  = 1.0 - min(report["color_deviation"] / 3.0, 1.0)   # تقريبي

        # أوزان معقولة (يمكن تعديلها لاحقاً)
        score = (
            ssim_norm   * 35 +
            lpips_norm  * 35 +
            detail_norm * 15 +
            color_norm  * 15
        )

        report["overall_quality_score"] = round(score, 2)

        # ── تحديد المشاكل ─────────────────────────────────────────────────
        if report["ssim_score"] is not None and report["ssim_score"] < min_ssim_threshold:
            report["issues"].append(f"SSIM منخفض ({report['ssim_score']:.4f})")

        if report["lpips_score"] is not None and report["lpips_score"] > max_lpips_threshold:
            report["issues"].append(f"LPIPS مرتفع ({report['lpips_score']:.4f})")

        if report["detail_loss_ratio"] > detail_loss_threshold:
            report["issues"].append(f"فقدان تفاصيل ({report['detail_loss_ratio']:.4f})")

        if report["color_deviation"] > color_dev_threshold:
            report["issues"].append(f"انحراف ألوان ({report['color_deviation']:.4f})")

        report["needs_correction"] = len(report["issues"]) > 0

        return report


    def post_monitor_filter(
        self,
        monitoring_report: Dict[str, Any],
        original_prompt: str,
        current_prompt: str,
        max_iterations_allowed: int = 2,
    ) -> Tuple[str, bool, List[str]]:
        """
        توليد prompt تصحيحي بناءً على تقرير الـ monitoring.
        يراعي عدد التكرارات لمنع الحلقات اللا نهائية.
        """
        issues = monitoring_report.get("issues", [])
        needs_repair = monitoring_report.get("needs_correction", False)

        if not needs_repair:
            return current_prompt, False, []

        # لو وصلنا للحد الأقصى → نرجع بدون تعديل ونوقف
        current_iter = monitoring_report.get("iteration_count", 0)  # افتراضي 0 لو مش موجود
        if current_iter >= max_iterations_allowed:
            return current_prompt, False, ["تم الوصول للحد الأقصى لمحاولات التصحيح"]

        correction_additions = []
        reasons = []

        # 1. هيكل / تشوهات
        if (
            monitoring_report.get("ssim_score", 1.0) < 0.92
            or monitoring_report.get("detail_loss_ratio", 0.0) > 0.08
            or monitoring_report.get("lpips_score", 0.0) > 0.30
        ):
            correction_additions.append(
                "perfect anatomy, correct proportions, sharp structural details, "
                "no deformation, high fidelity to original pose and geometry"
            )
            reasons.append("مشكلة هيكلية أو فقدان تفاصيل أو perceptual difference")

        # 2. ألوان / توازن
        if monitoring_report.get("color_deviation", 0.0) > 0.15:
            correction_additions.append(
                "natural balanced colors, no color bleeding or cast, realistic saturation"
            )
            reasons.append("انحراف ألوان ملحوظ")

        # 3. جودة عامة / إدراكية
        if (
            "جودة إدراكية منخفضة" in issues
            or monitoring_report.get("overall_quality_score", 100) < 65
        ):
            correction_additions.append(
                "masterpiece, ultra detailed, cinematic quality, high resolution"
            )
            reasons.append("جودة إدراكية أو كلية منخفضة")

        # إذا ما حصلش أي تصحيح → نرجع بدون تغيير
        if not correction_additions:
            return current_prompt, False, []

        # دمج بطريقة أنظف
        additions_str = ", ".join(correction_additions)
        new_prompt = f"{current_prompt.strip()}, {additions_str}".strip(", ")

        # تنظيف نهائي (أكثر شمولاً)
        new_prompt = re.sub(r'\s+', ' ', new_prompt)
        new_prompt = re.sub(r'\s*,\s*', ', ', new_prompt)
        new_prompt = re.sub(r'^,\s*|\s*,$', '', new_prompt).strip()

        return new_prompt, True, reasons


    def finalize_output(
        self,
        final_image: Image.Image,
        original_image: Image.Image,
        monitoring_report: Dict[str, Any],
        dna_genes: Dict[str, str],
        final_prompt: str,
        output_path: str = "dna_repaired_final.jpg",
        report_path: str = "dna_repair_report.json",
        sharpen: float = 1.12,
        contrast: float = 1.06,
        color: float = 1.04,
        start_time: Optional[float] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        الخطوة النهائية: تلميع + تقرير + حفظ (آمن ومنظم)
        """
        # 1. تلميع نهائي (مع تحويل آمن)
        polished = final_image.convert("RGB")
        polished = ImageEnhance.Sharpness(polished).enhance(sharpen)
        polished = ImageEnhance.Contrast(polished).enhance(contrast)
        polished = ImageEnhance.Color(polished).enhance(color)

        # 2. حساب الوقت (مع fallback واضح)
        end_time = datetime.now().timestamp()
        duration = round(end_time - (start_time or end_time), 2)
        if start_time is None:
            duration_note = "غير متوفر (بدون start_time)"
        else:
            duration_note = f"{duration:.2f} ثانية"

        # 3. إنشاء التقرير النهائي
        full_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "execution_duration": duration_note,
            "original_prompt": final_prompt,
            "final_prompt_used": final_prompt,
            "dna_genes": dna_genes,
            "monitoring": monitoring_report,
            "final_quality_score": round(monitoring_report.get("overall_quality_score", 0), 1),
            "status": "success" if not monitoring_report.get("needs_correction", False) else "partial",
            "issues": monitoring_report.get("issues", []),
            "output_path": output_path,
            "report_path": report_path,
            "image_size": f"{final_image.size[0]}x{final_image.size[1]}",
        }

        # 4. حفظ الصورة + التقرير (مع try/except)
        try:
            polished.save(output_path, quality=95)
            print(f"تم حفظ الصورة: {output_path}")
        except Exception as e:
            print(f"خطأ في حفظ الصورة: {e}")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(full_report, f, ensure_ascii=False, indent=2)
            print(f"تم حفظ التقرير: {report_path}")
        except Exception as e:
            print(f"خطأ في حفظ التقرير: {e}")

        return polished, full_report


    def run_full_pipeline(self):
        """Method رئيسي لتشغيل كل الخطوات تلقائيًا"""
        self.get_user_input()
        self.filter_and_enhance_prompt(self.current_prompt)
        self.extract_initial_dna_genes(self.current_prompt, self.current_image)
        # ... استدعاء باقي الـ methods بالترتيب
        self.finalize_output(...)

# ────────────────────────────────────────────────
# اختبار سريع
# ────────────────────────────────────────────────
if __name__ == "__main__":
    test_prompt = "وجه بنت جميلة في غابة سحرية، إضاءة ذهبية، أجواء غامضة، blurry, low quality"
    filtered, genes = prepare_prompt_and_genes(raw_prompt=test_prompt)
    print("Filtered:", filtered)
    print("Genes:", genes)

    repair = DNARepairPipeline()

    test_prompt = "وجه بنت جميلة في غابة سحرية، إضاءة ذهبية، أجواء غامضة"
    genes_initial = repair.extract_initial_dna_genes(test_prompt)
    genes_scene = repair.extract_scene_dna_genes(test_prompt)
    print("Initial Genes:", genes_initial)
    print("Scene Genes:", genes_scene)
