"""3AM Tales GPU engine on Modal (free $30/month of GPU time on Modal's Starter plan).

Two open models, loaded once per run into an H100:
- QwenEdit : Qwen-Image-Edit-2509 + 8-step Lightning LoRA. Draws a shot from the characters' portraits, so faces
             and outfits stay the same in every picture (up to 3 reference images).
- WanI2V   : Wan 2.2 image-to-video (A14B) + 4-step lightx2v LoRA. Turns a picture into a ~5 s clip where the
             characters really move (480x832, 16 fps; the editor upscales it).
- download : fetches the model files once into a Modal Volume on a cheap CPU machine (no GPU time wasted).

Deployed automatically by the GitHub workflow (`modal deploy gpu_app.py`); main.py calls it through agents/gpu.py.
Every call is optional: if anything here fails, the reel is made the normal way.
"""
import io
import math

import modal

APP_NAME = "3am-tales-gpu"
QWEN_EDIT = "Qwen/Qwen-Image-Edit-2509"
QWEN_LORA_REPO = "lightx2v/Qwen-Image-Lightning"
QWEN_LORA_FILE = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
WAN_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_LORA_REPO = "Kijai/WanVideo_comfy"
WAN_LORA_FILE = "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

app = modal.App(APP_NAME)
models = modal.Volume.from_name("3am-tales-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "diffusers>=0.36.0", "transformers>=4.56,<5",
                 "accelerate>=1.10", "peft>=0.17", "safetensors", "sentencepiece", "ftfy", "Pillow",
                 "imageio[ffmpeg]", "huggingface_hub[hf_transfer]>=0.34")
    .env({"HF_HOME": "/models/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
VOL = {"/models": models}


@app.function(image=image, volumes=VOL, timeout=3 * 3600, cpu=4, memory=8192)
def download() -> str:
    """One-time (and cheap) download of all model files into the volume. Safe to call again: it skips files
    that are already there."""
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download(QWEN_EDIT)
    hf_hub_download(QWEN_LORA_REPO, QWEN_LORA_FILE)
    snapshot_download(WAN_I2V)
    hf_hub_download(WAN_LORA_REPO, WAN_LORA_FILE)
    models.commit()
    return "models ready"


@app.cls(image=image, volumes=VOL, gpu="H100", timeout=3600, scaledown_window=90)
class QwenEdit:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        sched = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
            "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0,
            "shift_terminal": None, "stochastic_sampling": False, "time_shift_type": "exponential",
            "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False,
            "use_karras_sigmas": False})
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN_EDIT, scheduler=sched,
                                                              torch_dtype=torch.bfloat16).to("cuda")
        try:
            self.pipe.load_lora_weights(QWEN_LORA_REPO, weight_name=QWEN_LORA_FILE)
            self.pipe.fuse_lora()
            self.steps, self.cfg = 8, 1.0
        except Exception as e:  # noqa: BLE001 — without the fast LoRA it still works, just slower
            print(f"Lightning LoRA not loaded ({e}); using 30 normal steps")
            self.steps, self.cfg = 30, 4.0

    @modal.method()
    def draw(self, prompt: str, refs: list[bytes], width: int, height: int, seed: int) -> bytes:
        import torch
        from PIL import Image
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in refs[:3]]
        out = self.pipe(image=imgs, prompt=prompt, negative_prompt="text, watermark, deformed, extra fingers",
                        height=height, width=width, num_inference_steps=self.steps, true_cfg_scale=self.cfg,
                        guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()


@app.cls(image=image, volumes=VOL, gpu="H100", timeout=3600, scaledown_window=90)
class WanI2V:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import WanImageToVideoPipeline
        self.pipe = WanImageToVideoPipeline.from_pretrained(WAN_I2V, torch_dtype=torch.bfloat16)
        try:
            self.pipe.load_lora_weights(WAN_LORA_REPO, weight_name=WAN_LORA_FILE, adapter_name="fast")
            self.pipe.load_lora_weights(WAN_LORA_REPO, weight_name=WAN_LORA_FILE, adapter_name="fast_2",
                                        load_into_transformer_2=True)
            self.pipe.set_adapters(["fast", "fast_2"], adapter_weights=[1.0, 1.0])
            self.steps, self.cfg = 4, 1.0
        except Exception as e:  # noqa: BLE001
            print(f"lightx2v LoRA not loaded ({e}); using 30 normal steps")
            self.steps, self.cfg = 30, 3.5
        self.pipe.enable_model_cpu_offload()     # the two 14B experts take turns on the GPU

    @modal.method()
    def animate(self, picture: bytes, prompt: str, seed: int, frames: int = 81) -> bytes:
        import tempfile

        import torch
        from diffusers.utils import export_to_video
        from PIL import Image
        img = Image.open(io.BytesIO(picture)).convert("RGB").resize((480, 832), Image.LANCZOS)
        video = self.pipe(image=img, prompt=prompt,
                          negative_prompt="static, frozen, blurry, deformed face, extra limbs, text, watermark",
                          height=832, width=480, num_frames=frames, num_inference_steps=self.steps,
                          guidance_scale=self.cfg, guidance_scale_2=self.cfg,
                          generator=torch.Generator("cuda").manual_seed(seed)).frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            export_to_video(video, f.name, fps=16)
            return open(f.name, "rb").read()
