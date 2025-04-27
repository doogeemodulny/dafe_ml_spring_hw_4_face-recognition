from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from train import FaceNetModel, get_test_transforms


def debug_show(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Class: {image_path.split('/')[5]}")
    plt.show()


def image_to_tensor(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    transformed = transform(image=image)
    tensor = transformed['image']
    return tensor


def get_embedding(image_tensor, model):
    with torch.no_grad():
        embedding = model(image_tensor.unsqueeze(0))
    return embedding


def evaluate_recognition(model, test_images, transform, threshold=0.7):
    """Оценивает модель на тестовых изображениях."""

    model.eval()
    names = set([x.split("/")[1] for x in test_images])
    for name in names:
        if name != "edik":
            continue
        name_images = [x for x in test_images if name in x]
        other_images = [x for x in test_images if name not in x]

        reference_tensor = image_to_tensor(name_images[0], transform=transform)
        reference_embedding = get_embedding(reference_tensor, model)

        correct_accepts = 0
        correct_rejects = 0
        false_accepts = 0
        false_rejects = 0

        for idx in range(1, len(name_images)):
            test_tensor = image_to_tensor(name_images[idx], transform=transform)
            test_embedding = get_embedding(test_tensor, model)

            similarity = F.cosine_similarity(reference_embedding, test_embedding)
            similarity_value = similarity.item()

            if similarity_value > threshold:
                correct_accepts += 1
            if similarity_value <= threshold:
                false_rejects += 1

        for idx in range(len(other_images)):
            test_tensor = image_to_tensor(other_images[idx], transform=transform)
            test_embedding = get_embedding(test_tensor, model)

            similarity = F.cosine_similarity(reference_embedding, test_embedding)
            similarity_value = similarity.item()

            if similarity_value > threshold:
                false_accepts += 1
            if similarity_value <= threshold:
                correct_rejects += 1

        print(name)
        print(f"Правильно распознаны {correct_accepts} из {correct_accepts + false_rejects}")
        print(f"Правильно отклонены {correct_rejects} из {false_accepts + correct_rejects}")


def evaluate_similarity(model, im1, im2):
    _tensor_1 = image_to_tensor(im1, transform=get_test_transforms())
    _tensor_2 = image_to_tensor(im2, transform=get_test_transforms())
    _embedding_1 = get_embedding(_tensor_1, model)
    _embedding_2 = get_embedding(_tensor_2, model)

    similarity = F.cosine_similarity(_embedding_1, _embedding_2)
    similarity_value = similarity.item()
    return similarity_value


if __name__ == "__main__":
    model = FaceNetModel()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_folders = [
        "test_images/edik/",
        "test_images/nikita/",
        "test_images/ildar/",
        "test_images/ernest/"
    ]

    # Собираем полные пути к изображениям
    test_images = []
    for folder in test_folders:
        file_names = os.listdir(folder)
        full_paths = [os.path.join(folder, name) for name in file_names]
        test_images.extend(full_paths)
    evaluate_recognition(model, test_images, transform=get_test_transforms(), threshold=0.75)