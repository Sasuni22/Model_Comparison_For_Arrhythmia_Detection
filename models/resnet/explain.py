import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.resnet.model import ResNet1D
from models.resnet.gradcam import GradCAM1D
from utils.data_loader import get_dataloaders


def get_args():
    parser = argparse.ArgumentParser(description='Grad-CAM for 1D ResNet')
    parser.add_argument('--signals', type=str, default='data/MITBIH/processed/signals.npy')
    parser.add_argument('--labels', type=str, default='data/MITBIH/processed/labels.npy')
    parser.add_argument('--ckpt', type=str, default='models/resnet/saved_model/best_model.pth')
    parser.add_argument('--num_samples', type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, seg_len, num_classes = get_dataloaders(
        signals_path=args.signals,
        labels_path=args.labels,
        batch_size=1,
        num_workers=0,
    )

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ResNet1D(num_classes=ckpt['num_classes'],
                     input_length=ckpt['seg_len']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Target layer = last residual stage
    target_layer = model.stage3[-1].conv2

    gradcam = GradCAM1D(model, target_layer)

    os.makedirs("gradcam_outputs", exist_ok=True)

    for i, (X, y) in enumerate(test_loader):
        if i >= args.num_samples:
            break

        X = X.to(device)

        output = model(X)
        pred = output.argmax(1).item()
        confidence = torch.softmax(output, 1)[0, pred].item()

        cam = gradcam.generate(X)

        signal = X.cpu().numpy()[0][0]
        cam = cam.numpy()[0]

        # Resize CAM to match signal length
        cam = np.interp(
            np.linspace(0, len(cam), len(signal)),
            np.arange(len(cam)),
            cam
        )

        plt.figure(figsize=(10, 4))
        plt.plot(signal, label='ECG Signal')
        plt.plot(cam * signal.max(), alpha=0.5, label='Grad-CAM')
        plt.title(f"True: {y.item()} | Pred: {pred} | Conf: {confidence:.2f}")
        plt.legend()
        plt.savefig(f"gradcam_outputs/sample_{i}.png")
        plt.close()

        print("="*50)
        print(f"Sample {i}")
        print(f"True Label: {y.item()}")
        print(f"Predicted : {pred}")
        print(f"Confidence: {confidence:.4f}")

    print("\nGrad-CAM explanations saved to gradcam_outputs/")


if __name__ == '__main__':
    main()