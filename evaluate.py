
import argparse, os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
from sklearn.metrics import (classification_report,
                             confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

def build_loader(data_dir, batch_size=32, num_workers=2):
    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=tfms)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    return loader, ds.classes

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, class_names = build_loader(args.data_dir,
                                       args.batch_size, args.num_workers)

    model = timm.create_model('pvt_v2_b0',
                              pretrained=False,
                              num_classes=len(class_names))
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu()
        all_preds.extend(preds)
        all_labels.extend(y)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Per‑class accuracy ──────────────────────────────────────────────────────
    correct = np.zeros(len(class_names))
    total   = np.zeros(len(class_names))
    for p, t in zip(all_preds, all_labels):
        total[t]   += 1
        correct[t] += (p == t)

    print("\nPer‑class accuracy:")
    for idx, cname in enumerate(class_names):
        acc = 100 * correct[idx] / total[idx]
        print(f"  {cname:20s} : {acc:6.2f}%")

    print(f"\nOverall accuracy: {100 * correct.sum() / total.sum():.2f}%\n")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm,
                annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ── Full classification report ─────────────────────────────────────────────
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset", type=str)
    p.add_argument("--weights_path", default="best_pvt_model.pth")
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    main(p.parse_args())
