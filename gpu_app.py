"""Darawni Kahani GPU engine on Modal (Starter plan: $30/month of free GPU time).

One open model from a non-Chinese lab, loaded once per run into an H100:
- LTXI2V : Lightricks LTX-Video 0.9.7 distilled (Israel, open weights) image-to-video. Turns a drawn picture into a
           ~4 s clip where the scene really moves (a ghost turning its head, hair and dupatta in the wind, mist
           drifting, a lantern flickering) instead of a camera move over a still picture.
- download : fetches the model files once into a Modal Volume on a cheap CPU machine (no GPU time wasted).

Deployed automatically by the GitHub workflow (`modal deploy gpu_app.py`); main.py calls it through agents/gpu.py.
Every call is optional: if anything here fails, the reel is made the normal way (camera moves).
"""
import io

import modal

APP_NAME = "3am-tales-gpu"
LTX_I2V = "Lightricks/LTX-Video-0.9.7-distilled"
# the distilled model's own 8-step schedule (from the model card)
LTX_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]

app = modal.App(APP_NAME)
models = modal.Volume.from_name("3am-tales-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "diffusers>=0.36.0", "transformers>=4.56,<5",
                 "accelerate>=1.10", "safetensors", "sentencepiece", "ftfy", "Pillow",
                 "imageio[ffmpeg]", "huggingface_hub[hf_transfer]>=0.34")
    .env({"HF_HOME": "/models/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
VOL = {"/models": models}


@app.function(image=image, volumes=VOL, timeout=3 * 3600, cpu=4, memory=8192)
def download() -> str:
    """One-time (and cheap) download of the model files into the volume. Safe to call again."""
    from huggingface_hub import snapshot_download
    snapshot_download(LTX_I2V)
    models.commit()
    return "models ready"


@app.cls(image=image, volumes=VOL, gpu="H100", timeout=3600, scaledown_window=90)
class LTXI2V:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import LTXConditionPipeline
        self.pipe = LTXConditionPipeline.from_pretrained(LTX_I2V, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.vae.enable_tiling()

    @modal.method()
    def animate(self, picture: bytes, prompt: str, seed: int, frames: int = 97) -> bytes:
        """picture → mp4 (512x896 portrait, 24 fps, ~4 s). frames must be 8k+1."""
        import tempfile

        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import export_to_video
        from PIL import Image
        w, h = 512, 896                                   # divisible by 32, close to 9:16
        img = Image.open(io.BytesIO(picture)).convert("RGB").resize((w, h), Image.LANCZOS)
        frames = max(9, (int(frames) - 1) // 8 * 8 + 1)
        args = dict(conditions=[LTXVideoCondition(image=img, frame_index=0)], prompt=prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, "
                                    "extra limbs, morphing, text, watermark, static",
                    width=w, height=h, num_frames=frames, guidance_scale=1.0, decode_timestep=0.05,
                    decode_noise_scale=0.025, image_cond_noise_scale=0.0,
                    generator=torch.Generator("cuda").manual_seed(seed))
        try:
            video = self.pipe(**args, timesteps=LTX_TIMESTEPS).frames[0]
        except TypeError:                                 # older/newer diffusers without custom timesteps
            video = self.pipe(**args, num_inference_steps=8).frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            export_to_video(video, f.name, fps=24)
            return open(f.name, "rb").read()
