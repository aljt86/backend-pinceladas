import cv2
import numpy as np

class EmotionClassifier:
    def __init__(self):
        # Lista de emociones que vamos a manejar
        self.emotions = ["alegría", "tristeza", "miedo", "enojo", "calma", "sorpresa"]

    def predict_emotion(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Simulación: generar probabilidades aleatorias que sumen 1
        raw_scores = np.random.rand(len(self.emotions))
        probabilities = raw_scores / np.sum(raw_scores)

        # Seleccionar la emoción con mayor probabilidad
        max_index = int(np.argmax(probabilities))
        predicted_emotion = self.emotions[max_index]
        confidence = float(probabilities[max_index])

        # Construir diccionario con todas las predicciones
        all_predictions = {
            emo: float(prob) for emo, prob in zip(self.emotions, probabilities)
        }

        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "all_predictions": all_predictions
        }