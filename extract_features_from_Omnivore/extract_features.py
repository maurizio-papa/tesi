import torch

device = "cuda" if torch.cuda.is_available() else "cpu" 
model_name = "omnivore_swinB"
model = torch.hub.load("facebookresearch/omnivore:main", model=model_name, force_reload=True)
model = model.to(device)
model = model.eval()


# The model expects inputs of shape: B x C x T x H x W
video_input = video_inputs[0][None, ...]

with torch.no_grad():
    prediction = model(video_input.to(device), input_type="video")

