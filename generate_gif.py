from PIL import Image
from qlearning import escape
import torch
import imageio
from PIL import Image


device = torch.device("cuda:0")
je = escape.JourneyEscape(device=device)
frames = je.play()
# frames = [Image.fromarray(frame) for frame in frames]
output_path = "assets/journey_escape.gif"

imageio.mimsave(output_path, frames, 'GIF', duration=1/len(frames))
