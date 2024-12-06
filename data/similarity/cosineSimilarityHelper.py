import torch
import tensorflow as tf


def fetchImageFromPath(path):
    image_loaded = tf.keras.utils.load_img(path)
    image_loaded = tf.image.resize(images=image_loaded, size=(256, 256))
    return image_loaded


class CosineSimilarityCalculator:
    def __init__(self, image_dataset, image_paths) -> None:
        self.image_dataset = image_dataset
        self.image_paths = image_paths

    def getCosineSimilarityFromImages(self, i) -> None:
        for j in range(i+1, len(self.image_dataset)):
            tensor_i = torch.from_numpy(self.image_dataset[i]).reshape(1, -1).squeeze()
            tensor_j = torch.from_numpy(self.image_dataset[j]).reshape(1, -1).squeeze()
            cos_similarity = float(torch.nn.CosineSimilarity(dim=0)(tensor_i, tensor_j))
            print(
                str(cos_similarity) +
                "," + 
                self.image_paths[i] +
                "," +
                self.image_paths[j]
            )
