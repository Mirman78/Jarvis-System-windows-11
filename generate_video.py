import torch
import torch_directml
from diffusers import StableVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
import os

def generate_video(input_image_path, output_video_path="output.mp4"):
    """
    Generates a short video from an image using Stable Video Diffusion.
    Optimized for AMD GPUs via DirectML.
    """
    try:
        # Use DirectML device for AMD GPU
        device = torch_directml.device()
        dtype = torch.float32 # DirectML usually prefers float32 for stability

        print(f"Using device: {device}")

        # Load pipeline
        pipeline = StableVideoPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-v1",
            torch_dtype=dtype
        ).to(device)

        # Load and prepare image
        image = Image.open(input_image_path).convert("RGB")
        image = image.resize((1024, 576)) # SVD expects specific dimensions

        print("Generating frames... (this may take a few minutes on AMD 580)")
        generator = torch.manual_seed(42)
        
        # decode_chunk_size=2 helps fit in 8GB VRAM
        frames = pipeline(image, decode_chunk_size=2, generator=generator).frames[0]

        # Export to video
        export_to_video(frames, output_video_path, fps=7)
        
        print(f"✅ Video successfully saved to: {output_video_path}")
        return True

    except Exception as e:
        print(f"❌ Error generating video: {e}")
        return False

if __name__ == "__main__":
    input_path = r"C:\Users\ME\Desktop\chat.png"
    if os.path.exists(input_path):
        generate_video(input_path)
    else:
        print(f"Input image not found at {input_path}")
