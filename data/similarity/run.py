import numpy as np
import tensorflow as tf
import torch
from PIL import UnidentifiedImageError


def getImageDatasetFromPaths(paths):
    image_dataset = []
    image_paths = []

    for path in paths:
        image_dataset.append(fetchImageFromPath(path))
        image_paths.append(path)

    return np.array(image_dataset), np.array(image_paths)


def fetchImageFromPath(path):
    image_loaded = None
    try:
        image_loaded = tf.keras.utils.load_img(path)
        image_loaded = tf.image.resize(images=image_loaded, size=(256, 256))
    except UnidentifiedImageError:
        print(path)

    return image_loaded


def getCosineSimilarityFromImages(image_dataset, image_paths):
    for i in range(len(image_dataset)):
        for j in range(i + 1, len(image_dataset)):
            tensor_i = torch.from_numpy(image_dataset[i]).reshape(1, -1).squeeze()
            tensor_j = torch.from_numpy(image_dataset[j]).reshape(1, -1).squeeze()
            cos_similarity = float(torch.nn.CosineSimilarity(dim=0)(tensor_i, tensor_j))
            print(str(cos_similarity) + "," + image_paths[i] + "," + image_paths[j])


with open("paths.txt", "r") as file:
    paths = file.readlines()
    paths = [i.rstrip() for i in paths]

    image_dataset, image_paths = getImageDatasetFromPaths(paths)
    getCosineSimilarityFromImages(image_dataset, image_paths)

