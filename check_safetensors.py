from safetensors import safe_open
import torch

file_path = r"C:\Users\Rashed_Dadou\Desktop\SuperVisorSmartReporter\Rehabilitation Pipeline\update\Update\diffusion_pytorch_model.safetensors"

try:
    with safe_open(file_path, framework="pt", device="cpu") as f:
        print(f"الملف سليم ✓")
        print(f"عدد المفاتيح (layers): {len(f.keys())}")
        print("\nأول 10 مفاتيح كمثال:")
        for key in list(f.keys())[:10]:
            print("   ", key)

        # مثال: شكل وزن معين (لو موجود)
        example_key = "down_blocks.0.attentions.0.proj_in.weight"
        if example_key in f.keys():
            tensor = f.get_tensor(example_key)
            print(f"\nشكل الـ tensor '{example_key}': {tensor.shape}")
            print(f"نوعه: {tensor.dtype}")
            print(f"أول 5 قيم: {tensor.flatten()[:5]}")

except Exception as e:
    print("مشكلة في فتح الملف:")
    print(e)
