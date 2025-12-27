import os
import sys
import shutil
import subprocess
import argparse
import re
import csv
import io

# Configuration Paths
ENGINE_DIR = "./pretrained_weights/tensorrt"
ACTIVE_ENGINE_NAME = "unet_work.engine"
BUILD_SCRIPT = "torch2trt.py"
INFERENCE_SCRIPT = "inference_online.py"

def get_gpus_from_nvidia_smi():
    """
    Lists GPUs using nvidia-smi to avoid PyTorch initialization crashes.
    Returns a list of dicts: [{'index': 0, 'name': 'RTX 3090', 'uuid': ...}]
    """
    try:
        # Query nvidia-smi for index, name, and total memory
        cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        gpus = []
        reader = csv.reader(io.StringIO(result.stdout))
        for row in reader:
            if len(row) < 3: continue
            gpus.append({
                "index": int(row[0].strip()),
                "name": row[1].strip(),
                "memory": row[2].strip()
            })
        return gpus
    except Exception as e:
        print(f"⚠️  Failed to query nvidia-smi: {e}")
        return []

def get_sanitized_name(name):
    """Returns a filesystem-safe name."""
    return re.sub(r'[^a-zA-Z0-9]', '_', name)

def manage_engine(gpu_info):
    """
    Ensures the correct engine file is in place for the selected GPU.
    """
    gpu_name = get_sanitized_name(gpu_info['name'])
    cached_engine_filename = f"unet_work_{gpu_name}.engine"
    
    active_engine_path = os.path.join(ENGINE_DIR, ACTIVE_ENGINE_NAME)
    cached_engine_path = os.path.join(ENGINE_DIR, cached_engine_filename)

    # 1. Check if we already have a compiled engine for this specific GPU
    if os.path.exists(cached_engine_path):
        print(f"✅ Found cached engine for {gpu_info['name']}.")
        print(f"   Restoring from: {cached_engine_filename}")
        shutil.copy2(cached_engine_path, active_engine_path)
        return True
    
    # 2. If not, we need to build it.
    print(f"⚠️  No engine found for {gpu_info['name']}.")
    print(f"🚀 Starting compilation via {BUILD_SCRIPT}...")
    print("   (This may take 10-20 minutes. Please wait.)\n")

    # Set env var so the build script sees ONLY the selected GPU as 'cuda:0'
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_info['index'])

    try:
        # We run the build script as a subprocess
        subprocess.run([sys.executable, BUILD_SCRIPT], env=env, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Engine compilation failed!")
        return False

    # 3. Cache the result for next time
    if os.path.exists(active_engine_path):
        print(f"\n💾 Caching new engine to: {cached_engine_filename}")
        shutil.copy2(active_engine_path, cached_engine_path)
        return True
    else:
        print(f"\n❌ Build finished but {ACTIVE_ENGINE_NAME} was not found.")
        return False

def run_inference(gpu_index, extra_args):
    """Launches the main application on the selected GPU."""
    print(f"\n🟢 Launching {INFERENCE_SCRIPT} on GPU {gpu_index}...")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    
    cmd = [sys.executable, INFERENCE_SCRIPT] + extra_args
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        pass

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Launcher")
    parser.add_argument("--gpu", type=int, help="Index of the GPU to use")
    parser.add_argument("inference_args", nargs=argparse.REMAINDER, help="Arguments for inference_online.py")
    args = parser.parse_args()

    # 1. Detect GPUs via nvidia-smi
    gpus = get_gpus_from_nvidia_smi()
    if not gpus:
        print("No NVIDIA GPUs found (or nvidia-smi failed).")
        return

    # 2. Select GPU
    selected_gpu = None
    
    # If passed via command line
    if args.gpu is not None:
        for g in gpus:
            if g['index'] == args.gpu:
                selected_gpu = g
                break
        if not selected_gpu:
            print(f"Error: GPU index {args.gpu} not found.")
            return

    # Interactive selection
    if selected_gpu is None:
        print("\nDetected NVIDIA GPU(s):")
        for g in gpus:
            print(f"  [{g['index']}] {g['name']} ({g['memory']})")
        print("")
        
        if len(gpus) == 1:
            selected_gpu = gpus[0]
            print(f"Auto-selecting: {selected_gpu['name']}")
        else:
            while True:
                try:
                    idx = int(input("Select GPU index to use: "))
                    for g in gpus:
                        if g['index'] == idx:
                            selected_gpu = g
                            break
                    if selected_gpu: break
                    print("Invalid index.")
                except ValueError:
                    print("Please enter a number.")

    print(f"\nSelected: GPU {selected_gpu['index']} - {selected_gpu['name']}")

    # 3. Manage Engine (Swap or Compile)
    # The build script inside here will run with CUDA_VISIBLE_DEVICES set
    if not manage_engine(selected_gpu):
        sys.exit(1)

    # 4. Launch Application
    pass_args = args.inference_args
    if "--acceleration" not in pass_args:
        pass_args.extend(["--acceleration", "tensorrt"])

    run_inference(selected_gpu['index'], pass_args)

if __name__ == "__main__":
    main()
