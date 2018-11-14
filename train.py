from qlearning import escape
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
je = escape.JourneyEscape(device=device)
je.train()
