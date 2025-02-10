import tempfile
import torch
from torch import Tensor
from torchvision.io import write_video

def export_to_video(tensor: Tensor, fps: int = 10) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    write_video(path, (tensor.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8), fps=fps)
    return path
