import os
import csv
import pandas as pd
from procesamiento import DrawingAnalyzer

# Configuración
DATA_DIR = "data"
IMAGES_DIR = DATA_DIR  # tus imágenes están directamente en data/
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset.csv")

# Inicializar analizador
analyzer = DrawingAnalyzer()

# Cargar etiquetas
labels_df = pd.read_csv(LABELS_FILE)
labels_dict = dict(zip(labels_df["id"], labels_df["label"]))

# Preparar escritura
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "id", "label",
        "avg_hue", "avg_saturation", "avg_brightness",
        "num_contours", "stroke_complexity",
        *[f"dc_{i}_{c}" for i in range(5) for c in ["r", "g", "b"]]
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Recorrer imágenes
    for fname in os.listdir(IMAGES_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_id = os.path.splitext(fname)[0]
        label = labels_dict.get(image_id)
        if not label:
            print(f"⚠️ Imagen sin etiqueta: {image_id}")
            continue

        path = os.path.join(IMAGES_DIR, fname)
        color = analyzer.extract_color_features(path)
        shape = analyzer.extract_shape_features(path)
        dom = analyzer.extract_dominant_colors(path)

        row = {
            "id": image_id,
            "label": label,
            "avg_hue": color.get("average_hue", 0),
            "avg_saturation": color.get("average_saturation", 0),
            "avg_brightness": color.get("average_brightness", 0),
            "num_contours": shape.get("num_contours", 0),
            "stroke_complexity": shape.get("stroke_complexity", 0),
        }

        for i in range(5):
            r, g, b = dom[i] if i < len(dom) else (0, 0, 0)
            row[f"dc_{i}_r"] = r
            row[f"dc_{i}_g"] = g
            row[f"dc_{i}_b"] = b

        writer.writerow(row)

print(f"✅ Dataset generado: {OUTPUT_FILE}")