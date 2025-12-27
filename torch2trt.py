import os
import torch
from omegaconf import OmegaConf
import gc
from src.modeling.framed_models import unet_work
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.models.unet_3d_explicit_reference import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder
from src.models.pose_guider import PoseGuider

from src.modeling.onnx_export import export_onnx, handle_onnx_batch_norm, optimize_onnx
from polygraphy.backend.trt import engine_from_network, network_from_onnx_path, save_engine, CreateConfig
from polygraphy.logger import G_LOGGER
import tensorrt as trt
G_LOGGER.severity = G_LOGGER.VERBOSE

def map_device(device_or_str):
    return device_or_str if isinstance(device_or_str, torch.device) else torch.device(device_or_str)

# parameters
batch_size = 1
height = 256
width = 256
onnx_opset = 17
device = torch.device("cuda:0")
device = map_device(device)
#dtype = torch.float16
dtype = torch.float32
config_path = './configs/prompts/personalive_online.yaml'
cfg = OmegaConf.load(config_path)
onnx_path = cfg.onnx_path
onnx_opt_path = cfg.onnx_opt_path
tensorrt_target_model = cfg.tensorrt_target_model

infer_config = OmegaConf.load(cfg.inference_config)
sched_kwargs = OmegaConf.to_container(
    infer_config.noise_scheduler_kwargs
)

pose_guider = PoseGuider().to(device=device, dtype=dtype)
pose_guider_state_dict = torch.load(cfg.pose_guider_path, map_location="cpu")
pose_guider.load_state_dict(pose_guider_state_dict)
del pose_guider_state_dict

motion_encoder:MotEncoder = MotEncoder().to(dtype=dtype, device=device).eval()
motion_encoder.set_attn_processor(AttnProcessor())
motion_encoder_state_dict = torch.load(cfg.motion_encoder_path, map_location="cpu")
motion_encoder.load_state_dict(motion_encoder_state_dict)
del motion_encoder_state_dict

denoising_unet:UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
    cfg.pretrained_base_model_path,
    "",
    subfolder="unet",
    unet_additional_kwargs=infer_config.unet_additional_kwargs,
).to(dtype=dtype, device=device)

denoising_unet.load_state_dict(
    torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False
)
denoising_unet.load_state_dict(
    torch.load(
        cfg.temporal_module_path,
        map_location="cpu",
    ),
    strict=False,
)
denoising_unet.set_attn_processor(AttnProcessor())

vae:AutoencoderKL = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=device, dtype=dtype)
vae.set_default_attn_processor()

scheduler = DDIMScheduler(**sched_kwargs)
scheduler.to(device)
timesteps = torch.tensor([0,0,0,0,333,333,333,333,666,666,666,666,999,999,999,999], device=device).long()
scheduler.set_step_length(333)

model = unet_work(
    pose_guider,
    motion_encoder,
    denoising_unet,
    vae,
    scheduler,
    timesteps,
    )

if(not os.path.exists(os.path.dirname(onnx_path))):
    os.mkdir(os.path.dirname(onnx_path))

if not os.path.exists(onnx_path):
    export_onnx(
        model,
        onnx_path=onnx_path,
        opt_image_height=height,
        opt_image_width=width,
        opt_batch_size=batch_size,
        onnx_opset=onnx_opset,
        auto_cast=True,
        dtype=dtype,
        device=device,
    )

batch_size = 1
height = 512
width = 512
profile = model.get_dynamic_map(batch_size, height, width)
del model
gc.collect()
torch.cuda.empty_cache()

print('finished')

#print("Optimizing Onnx Model...")
#if(not os.path.exists(os.path.dirname(onnx_opt_path))):
#    os.mkdir(os.path.dirname(onnx_opt_path))
#optimize_onnx(
#    onnx_path=onnx_path,
#    onnx_opt_path=onnx_opt_path,
#)

engine = engine_from_network(
    network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]), #Old network_from_onnx_path(onnx_opt_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
    config=CreateConfig(
        fp16=True, refittable=False, profiles=[profile]
    ),
)
save_engine(engine, path=tensorrt_target_model)
gc.collect()
torch.cuda.empty_cache()
