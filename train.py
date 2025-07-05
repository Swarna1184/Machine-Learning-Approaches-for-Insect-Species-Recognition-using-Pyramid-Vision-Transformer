

import argparse, os, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def build_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # ImageFolder needs a transform per subset; switch val_ds to val_tfms
    val_ds.dataset.transform = val_tfms

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, full_ds.classes



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    preds, labels = [], []

    for inputs, y in loader:
        inputs, y = inputs.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds.extend(outputs.argmax(1).cpu())
        labels.extend(y.cpu())

    metrics = _calc_metrics(labels, preds)
    metrics["loss"] = epoch_loss / len(loader)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    preds, labels = [], []

    for inputs, y in loader:
        inputs, y = inputs.to(device), y.to(device)
        outputs = model(inputs)
        val_loss += criterion(outputs, y).item()
        preds.extend(outputs.argmax(1).cpu())
        labels.extend(y.cpu())

    metrics = _calc_metrics(labels, preds)
    metrics["loss"] = val_loss / len(loader)
    return metrics


def _calc_metrics(labels, preds):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='weighted', zero_division=0),
        "recall":    recall_score(labels, preds,  average='weighted', zero_division=0),
        "f1":        f1_score(labels, preds,      average='weighted', zero_division=0),
    }



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_dataloaders(args.data_dir,
                                                              args.batch_size,
                                                              args.num_workers)

    model = timm.create_model('pvt_v2_b0',
                              pretrained=True,
                              num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader,
                                        criterion, optimizer, device)
        val_metrics   = validate(model, val_loader,
                                 criterion, device)

        elapsed = int(time.time() - start)
        print(f"[{epoch:02}/{args.epochs}] "
              f"train_acc={train_metrics['accuracy']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"time={elapsed}s")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), "best_pvt_model.pth")
            print(f"  ✔ Saved new best model (val_acc={best_acc:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="dataset", type=str)
    p.add_argument("--epochs",     default=10,        type=int)
    p.add_argument("--batch_size", default=32,        type=int)
    p.add_argument("--lr",         default=1e-4,      type=float)
    p.add_argument("--num_workers",default=4,         type=int)
    main(p.parse_args())
