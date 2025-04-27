import os, sys, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_train_transforms():
    return A.Compose([
        A.Resize(height=128, width=128),
        A.HorizontalFlip(p=0.5),

        # Цветовые искажения
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ], p=0.5),

        # Размытие / шумы
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.NoOp(p=0.6),
        ], p=0.3),

        # Редкие аугментации
        A.ToGray(p=0.05),
        A.CoarseDropout(p=0.1),

        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def get_test_transforms():
    return A.Compose([
        A.Resize(height=128, width=128),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


class LFWDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label


# Модель для эмбеддингов
class FaceNetModel(nn.Module):
    def __init__(self, pretrained='vggface2', embedding_size=512):
        super(FaceNetModel, self).__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.last_linear.parameters():
            param.requires_grad = True
        for param in self.backbone.last_bn.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# Triplet Loss
class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def _get_anchor_positive_mask(self, labels):
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return labels_equal & indices_not_equal

    def _get_anchor_negative_mask(self, labels):
        return labels.unsqueeze(0) != labels.unsqueeze(1)

    def forward(self, embeddings, labels):
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        mask_anchor_positive = self._get_anchor_positive_mask(labels).float()
        mask_anchor_negative = self._get_anchor_negative_mask(labels).float()

        hardest_positive_dist = (pairwise_dist * mask_anchor_positive).max(dim=1)[0]

        negative_dist = pairwise_dist * mask_anchor_negative
        negative_dist[negative_dist == 0] = 10.0
        hardest_negative_dist = negative_dist.min(dim=1)[0]

        losses = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        return losses.mean()


def calculate_accuracy(embeddings, labels, threshold=1.0):
    with torch.no_grad():
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        mask_same_label = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        correct = ((pairwise_dist < threshold) == mask_same_label.bool()).float().mean()
    return correct.item()


def calculate_metrics(embeddings, labels, threshold=1.0):
    with torch.no_grad():
        n = len(labels)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))

        # Берем уникальные пары (i < j) для исключения дубликатов
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        dist_pairs = dist_matrix[triu_mask]
        same_label_pairs = same_label[triu_mask]

        pred_pairs = dist_pairs < threshold
        TP = (pred_pairs & same_label_pairs).sum().item()
        FP = (pred_pairs & ~same_label_pairs).sum().item()
        FN = (~pred_pairs & same_label_pairs).sum().item()
        accuracy = (pred_pairs == same_label_pairs).float().mean().item()
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

    return accuracy, precision, recall


if __name__ == "__main__":
    set_seed(42)

    data_dir = './lfw_data'
    os.makedirs(data_dir, exist_ok=True)
    lfw_dataset = fetch_lfw_people(data_home=data_dir, min_faces_per_person=30, download_if_missing=True, color=True)

    n_samples, h, w, _ = lfw_dataset.images.shape
    X = lfw_dataset.data
    n_features = X.shape[1]

    y = lfw_dataset.target
    target_names = lfw_dataset.target_names
    n_classes = target_names.shape[0]


    X_train, X_test, y_train, y_test = train_test_split(
        lfw_dataset.images, y, test_size=0.25, stratify=y, random_state=42)

    X_train = (X_train * 255).astype(np.uint8)
    X_test = (X_test * 255).astype(np.uint8)

    train_dataset = LFWDataset(X_train, y_train, transform=get_train_transforms())
    test_dataset = LFWDataset(X_test, y_test, transform=get_test_transforms())

    num_epochs = 10
    batch_size = 32
    lr = 0.001


    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FaceNetModel().to(device)
    criterion = OnlineTripletLoss(margin=0.5)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 1 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs} | '
                      f'Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                embeddings = model(images)
                loss = criterion(embeddings, labels)
                accuracy_batch, precision_batch, recall_batch = calculate_metrics(embeddings, labels, threshold=1.0)

                val_loss += loss.item()
                val_accuracy += accuracy_batch
                val_precision += precision_batch
                val_recall += recall_batch

        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)
        val_precision /= len(test_loader)
        val_recall /= len(test_loader)

        print(f'Epoch: {epoch + 1} | '
              f'Train Loss: {total_loss / len(train_loader):.4f} | '
              f'Test Loss: {val_loss:.4f} | '
              f'Test Accuracy: {val_accuracy:.4f} | '
              f'Test Precision: {val_precision:.4f} | '
              f'Test Recall: {val_recall:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    print('Training completed.')
