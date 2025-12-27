import os
import shutil
import site
import sys

def apply_patches():
    # 1. Locate the site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # --- Patch PyTorch ---
    torch_cuda_init = os.path.join(site_packages, "torch", "cuda", "__init__.py")
    my_patch = "patches/torch_cuda_init.py"
    
    if os.path.exists(torch_cuda_init) and os.path.exists(my_patch):
        print(f"Applying Blackwell/GB10 fix to: {torch_cuda_init}")
        shutil.copyfile(my_patch, torch_cuda_init)
    else:
        print("❌ Could not find PyTorch installation or patch file.")

    # --- Patch ONNX GraphSurgeon (The lambda fix) ---
    onnx_exporter = os.path.join(site_packages, "onnx_graphsurgeon", "exporters", "onnx_exporter.py")
    if os.path.exists(onnx_exporter):
        print(f"Patching ONNX GraphSurgeon at: {onnx_exporter}")
        with open(onnx_exporter, "r") as f:
            content = f.read()
        
        broken_ref = "onnx.helper.float32_to_bfloat16"
        fixed_code = "lambda x: (x.view(np.uint32) >> 16).astype(np.uint16)"
        
        if broken_ref in content:
            content = content.replace(broken_ref, fixed_code)
            with open(onnx_exporter, "w") as f:
                f.write(content)
            print("✅ ONNX patch applied.")
        else:
            print("ℹ️  ONNX file already patched or different version.")

if __name__ == "__main__":
    apply_patches()
