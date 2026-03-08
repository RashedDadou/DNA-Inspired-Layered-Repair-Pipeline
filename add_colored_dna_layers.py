"""
add_colored_dna_layers.py

الطبقة الخامسة: Colored DNA Layers
- وضع طبقات لونية مخصصة (دمج أخضر/أحمر/أزرق DNA-inspired)
- على أضلاع وشبكة الـ Net داخل حدود الماسك فقط
- دمج ذكي (density أو wave أو genetic_random)
- بدون طفرة عشوائية (دمج فقط كما طلبت)
"""

from typing import List, Literal, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2


def add_colored_dna_layers(
    net: Image.Image,                    # الشبكة الناتجة من ControlNet (Net)
    mask: Image.Image,                   # ماسك المنطقة المنهارة
    base_colors: List[Tuple[int, int, int]] = [
        (220, 60, 60),   # أحمر – طاقة / حيوية
        (60, 220, 60),   # أخضر – نمو / DNA أساسي
        (60, 60, 220)    # أزرق – توازن / برودة
    ],
    blend_mode: Literal["density", "wave", "genetic_random"] = "density",
    opacity_base: float = 0.52,          # شفافية أساسية (0.4–0.65 مثالي)
    edge_thickness: int = 4,             # سمك خطوط الشبكة
    edge_highlight_color: Tuple[int, int, int] = (200, 220, 255),  # لون فاتح للأضلاع
) -> Image.Image:
    """
    وضع طبقات لونية DNA-inspired على أضلاع وشبكة Net

    - يدمج الألوان الثلاثة بطريقة ذكية (بدون طفرة عشوائية)
    - يطبّق فقط داخل حدود الماسك
    - يرسم خطوط Net بلون فاتح لإبراز الهيكل

    Parameters:
        net: صورة الشبكة الناتجة من ControlNet
        mask: ماسك المنطقة (L mode)
        base_colors: قائمة الألوان الأساسية (أحمر/أخضر/أزرق)
        blend_mode:
            "density" → حسب كثافة الماسك (أحمر في المناطق القوية، أزرق في الضعيفة)
            "wave" → دمج موجي (DNA helix style)
            "genetic_random" → مزيج عشوائي جيني خفيف (لكن بدون طفرة قوية)
        opacity_base: شفافية الطبقة (تقل تدريجيًا حسب الـ blend)
        edge_thickness: سمك خطوط الشبكة
        edge_highlight_color: لون خطوط Net (فاتح للإبراز)

    Returns:
        PIL.Image: طبقة RGBA جاهزة للدمج مع الصورة
    """
    width, height = net.size
    net_arr = np.array(net.convert("RGBA"))
    mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

    # إنشاء طبقة جديدة شفافة
    layer = np.zeros((height, width, 4), dtype=np.uint8)

    # استخراج أضلاع الشبكة (edges)
    gray = cv2.cvtColor(net_arr[..., :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    if blend_mode == "density":
        # دمج حسب كثافة الماسك (أحمر في القوية، أزرق في الضعيفة)
        intensity = mask_arr ** 1.45
        for i, color in enumerate(base_colors):
            weight = intensity * (0.75 - i * 0.2)  # تدرج من أحمر إلى أزرق
            layer[..., :3] += (np.array(color) * weight[..., None]).astype(np.uint8)

    elif blend_mode == "wave":
        # دمج موجي (DNA helix style)
        h, w = mask_arr.shape
        wave = np.sin(np.linspace(0, 28 * np.pi, w) + np.arange(h)[:, None] * 0.06)
        wave = (wave + 1) / 2
        wave = np.repeat(wave[..., None], 3, axis=2)
        mixed = np.zeros((h, w, 3))
        for i, color in enumerate(base_colors):
            mixed += np.array(color) * wave * (1 / len(base_colors))
        layer[..., :3] = mixed.astype(np.uint8)

    elif blend_mode == "genetic_random":
        # دمج عشوائي جيني خفيف (DNA mixing)
        h, w = mask_arr.shape
        ratios = np.random.dirichlet([1, 1, 1], size=(h, w))  # نسب عشوائية متوازنة
        mixed = np.zeros((h, w, 3))
        for i, color in enumerate(base_colors):
            mixed += np.array(color) * ratios[..., i, None]
        layer[..., :3] = mixed.astype(np.uint8)

    # تطبيق الشفافية حسب الماسك
    layer[..., 3] = (mask_arr * 255 * opacity_base).astype(np.uint8)

    # رسم خطوط الشبكة (Net edges) بلون فاتح للإبراز
    layer[edges > 0, :3] = edge_highlight_color
    layer[edges > 0, 3] = 240  # شفافية عالية للخطوط

    return Image.fromarray(layer)


# ────────────────────────────────────────────────
# اختبار سريع (للتجربة المباشرة)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image

    # افتراض وجود net و mask
    net_img = Image.open("generated_net.jpg")
    mask_img = Image.open("mask.jpg").convert("L")

    colored_layers = add_colored_dna_layers(
        net=net_img,
        mask=mask_img,
        blend_mode="density",  # أو "wave" أو "genetic_random"
        opacity=0.52,
    )

    colored_layers.save("colored_dna_layers.jpg")
    print("تم إنشاء طبقات الألوان → colored_dna_layers.jpg")
