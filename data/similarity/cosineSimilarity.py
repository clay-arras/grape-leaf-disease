import numpy as np
import tensorflow as tf
import torch
from concurrent.futures import ProcessPoolExecutor
import cosineSimilarityHelper

MAX_WORKERS = 1
FILE_PATH = "paths.txt"


def getImageDatasetFromPaths(paths):
    image_dataset = []
    image_paths = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        images = executor.map(cosineSimilarityHelper.fetchImageFromPath, paths)
        for i, image in enumerate(images):
            image_dataset.append(image)
            image_paths.append(paths[i])

    return np.array(image_dataset), np.array(image_paths)


if __name__ == "__main__":
    with open(FILE_PATH, "r") as file:
        paths = file.readlines()
        paths = [i.rstrip() for i in paths]

        image_dataset, image_paths = getImageDatasetFromPaths(paths)
        similarityFn = cosineSimilarityHelper.CosineSimilarityCalculator(
            image_dataset=image_dataset, image_paths=image_paths
        )

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            id_mapping = [*range(len(image_dataset))]
            for i in executor.map(
                similarityFn.getCosineSimilarityFromImages, id_mapping
            ):
                pass

