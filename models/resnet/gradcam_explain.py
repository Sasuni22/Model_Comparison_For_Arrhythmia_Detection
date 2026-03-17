import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from models.resnet.model import ResNet1D
from utils.data_loader import get_dataloaders

# -----------------------------
# Load SNOMED CT mappings
# -----------------------------
with open("snomed_mapping.json") as f:
    snomed = json.load(f)

# -----------------------------
# Load model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet1D(num_classes=5)
checkpoint = torch.load("models/resnet/saved_model/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# -----------------------------
# Load test data
# -----------------------------
_, _, test_loader, _, _ = get_dataloaders(
    signals_path="data/MITBIH/processed/signals.npy",
    labels_path="data/MITBIH/processed/labels.npy",
    batch_size=1,
    num_workers=0
)

# -----------------------------
# Grad-CAM helper
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax().item()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx

# -----------------------------
# Target layer (last conv block)
# -----------------------------
target_layer = model.stage3[-1].conv2
gradcam = GradCAM(model, target_layer)

# -----------------------------
# Run Grad-CAM + SNOMED explanation
# -----------------------------
for idx, (x, y_true) in enumerate(test_loader):
    x = x.to(device)
    cam, class_idx = gradcam(x)
    pred_class = ['N', 'S', 'V', 'F', 'Q'][class_idx]
    snomed_info = snomed[pred_class]

    print(f"Predicted class: {pred_class} ({snomed_info['name']})")
    print(f"SNOMED CT Code: {snomed_info['code']}")
    print(f"Explanation: {snomed_info['description']}")

    # Plot ECG + Grad-CAM overlay
    plt.figure(figsize=(10, 3))
    plt.plot(x.cpu().squeeze().numpy(), label='ECG Signal')
    plt.plot(cam, color='r', alpha=0.5, label='Grad-CAM')
    plt.title(f"Prediction: {pred_class} ({snomed_info['name']})")
    plt.legend()
    plt.show()

    if idx == 4:  # demo first 5 signals only
        break