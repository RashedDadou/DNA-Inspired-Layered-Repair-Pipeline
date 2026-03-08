"""
prepare_scene_prompt_and_genes.py

الطبقة الأولى في النظام: Input Prompt + Filter Prompt + Scene DNA Generator
- ينقّي ويحسّن الـ Prompt الأولي
- يحلل النص + الصورة (اختياري) لاستخراج الجينات الأولية
- يرجع Prompt محسن + جينات المشهد (Scene DNA Genes)
"""

from typing import Dict, Optional, Tuple, List
import random
import json

# افتراض وجود مكتبات Vision LLM أو CLIP
# لو مش موجودة، نقدر نستبدل بـ templates بسيطة أو regex
try:
    from transformers import pipeline, CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# لو عندك LLM vision (LLaVA أو Qwen-VL أو GPT-4o vision API)، استخدمه هنا
# للبساطة، هنستخدم نمط template-based + CLIP كـ fallback


def filter_prompt(
    raw_prompt: str,
    style_preferences: Optional[List[str]] = None,
    quality_boosters: bool = True,
    negative_removal: bool = True,
) -> str:
    """
    فلتر الـ Prompt الأولي: تنقية + تحسين + توافق مع الـ DNA genes
    """
    filtered = raw_prompt.strip()

    # 1. إزالة كلمات سلبية شائعة (negative removal)
    if negative_removal:
        bad_words = ["blurry", "low quality", "ugly", "deformed", "bad anatomy", "extra limbs"]
        for word in bad_words:
            filtered = filtered.replace(word, "")

    # 2. إضافة معززات جودة (اختياري)
    if quality_boosters:
        boosters = [
            "masterpiece", "best quality", "highly detailed", "8k", "sharp focus",
            "cinematic lighting", "vibrant colors", "realistic details"
        ]
        filtered += ", " + ", ".join(boosters)

    # 3. إضافة تفضيلات أسلوبية (إذا موجودة)
    if style_preferences:
        filtered += ", " + ", ".join(style_preferences)

    # 4. تنظيف نهائي (إزالة مسافات زائدة، فواصل متكررة)
    filtered = " ".join(filtered.split())
    filtered = filtered.replace(" ,", ",").replace(",,", ",")

    return filtered.strip()


def extract_scene_dna_genes(
    prompt: str,
    image: Optional[Image.Image] = None,
    vision_model: Optional[str] = None,
) -> Dict[str, str]:
    """
    استخراج جينات المشهد (Scene DNA Genes) من الـ Prompt + الصورة (اختياري)
    """
    genes: Dict[str, str] = {}

    # 1. تحليل الـ Prompt بطريقة بسيطة (template-based)
    prompt_lower = prompt.lower()

    # pose_gene
    if any(word in prompt_lower for word in ["standing", "sitting", "lying", "dynamic", "action"]):
        genes["pose_gene"] = "dynamic pose" if "action" in prompt_lower else "static pose"
    else:
        genes["pose_gene"] = "neutral pose"

    # layout_gene
    if "centered" in prompt_lower or "symmetrical" in prompt_lower:
        genes["layout_gene"] = "centered symmetrical"
    elif "wide" in prompt_lower or "landscape" in prompt_lower:
        genes["layout_gene"] = "wide landscape"
    else:
        genes["layout_gene"] = "balanced composition"

    # style_gene
    style_keywords = ["photorealistic", "realistic", "cinematic", "anime", "oil painting", "digital art"]
    for keyword in style_keywords:
        if keyword in prompt_lower:
            genes["style_gene"] = keyword
            break
    else:
        genes["style_gene"] = "photorealistic"  # افتراضي

    # lighting_gene
    if any(word in prompt_lower for word in ["golden hour", "sunset", "dramatic", "volumetric", "god rays"]):
        genes["lighting_gene"] = "dramatic golden hour"
    elif "soft" in prompt_lower or "diffused" in prompt_lower:
        genes["lighting_gene"] = "soft diffused"
    else:
        genes["lighting_gene"] = "natural daylight"

    # camera_gene
    if any(word in prompt_lower for word in ["close-up", "portrait", "wide shot", "low angle", "aerial"]):
        genes["camera_gene"] = "close-up portrait" if "close" in prompt_lower else "wide shot"
    else:
        genes["camera_gene"] = "medium shot"

    # mood_gene
    mood_keywords = ["mysterious", "epic", "serene", "chaotic", "romantic", "dark", "hopeful"]
    for keyword in mood_keywords:
        if keyword in prompt_lower:
            genes["mood_gene"] = keyword
            break
    else:
        genes["mood_gene"] = "neutral"

    # color_seed_gene
    color_keywords = ["warm", "cool", "vibrant", "muted", "monochrome", "neon", "earthy"]
    for keyword in color_keywords:
        if keyword in prompt_lower:
            genes["color_seed_gene"] = keyword
            break
    else:
        genes["color_seed_gene"] = "vibrant natural"

    # لو في صورة + CLIP → تحسين الجينات (اختياري)
    if image is not None and CLIP_AVAILABLE:
        # هنا يمكن إضافة تحليل CLIP لتأكيد الجينات
        pass  # placeholder – يمكن توسيعه لاحقاً

    return genes


def prepare_scene_prompt_and_genes(
    raw_prompt: str,
    image: Optional[Image.Image] = None,
    style_preferences: Optional[List[str]] = None,
    quality_boosters: bool = True,
    negative_removal: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """
    الدالة الرئيسية للطبقة الأولى:
    1. Filter Prompt
    2. Scene DNA Genes
    """
    # 1. تنقية الـ Prompt
    filtered_prompt = filter_prompt(
        raw_prompt=raw_prompt,
        style_preferences=style_preferences,
        quality_boosters=quality_boosters,
        negative_removal=negative_removal,
    )

    # 2. استخراج الجينات
    genes = extract_scene_dna_genes(
        prompt=filtered_prompt,
        image=image,
    )

    return filtered_prompt, genes


# ────────────────────────────────────────────────
# اختبار سريع
# ────────────────────────────────────────────────

if __name__ == "__main__":
    prompt = "وجه واقعي جميل لفتاة في غابة سحرية، إضاءة ذهبية، أجواء غامضة"
    filtered, genes = prepare_scene_prompt_and_genes(prompt)

    print("Filtered Prompt:")
    print(filtered)
    print("\nScene DNA Genes:")
    print(json.dumps(genes, indent=2, ensure_ascii=False))
