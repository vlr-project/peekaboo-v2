import numpy as np
from PIL import Image

class ImageIoUCalculator:
    def __init__(self, image_path_1, image_path_2):
        self.image_array_1 = self.read_image_to_array(image_path_1)
        self.image_array_2 = self.read_image_to_array(image_path_2)

    @staticmethod
    def read_image_to_array(image_path):
        with Image.open(image_path) as img:
            image_array = np.array(img)
        return image_array

    @staticmethod
    def calculate_iou(matrix1, matrix2):
        # Ensure matrices are of the same size
        if matrix1.shape != matrix2.shape:
            raise ValueError("Both matrices must have the same dimensions")

        # Calculate Intersection and Union
        intersection = np.logical_and(matrix1, matrix2)
        union = np.logical_or(matrix1, matrix2)

        # Calculate IoU
        iou = np.sum(intersection) / np.sum(union)

        return iou

    def get_iou(self):
        return self.calculate_iou(self.image_array_1, self.image_array_2)