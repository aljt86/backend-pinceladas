import cv2
import numpy as np
from sklearn.cluster import KMeans

class DrawingAnalyzer:
    def __init__(self):
        pass

    def extract_color_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return {}

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_hue = float(np.mean(hsv[:, :, 0]))
        avg_sat = float(np.mean(hsv[:, :, 1])) / 255.0
        avg_val = float(np.mean(hsv[:, :, 2])) / 255.0

        return {
            "average_hue": round(avg_hue, 2),
            "average_saturation": round(avg_sat, 2),
            "average_brightness": round(avg_val, 2)
        }

    def extract_shape_features(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {}

        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        complexities = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area > 0:
                complexities.append(perimeter / area)
        stroke_complexity = float(np.mean(complexities)) if complexities else 0.0

        return {
            "num_contours": num_contours,
            "stroke_complexity": round(stroke_complexity, 2)
        }

    def extract_dominant_colors(self, image_path, k=5):
        image = cv2.imread(image_path)
        if image is None:
            return []

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)

        return [color.tolist() for color in colors]