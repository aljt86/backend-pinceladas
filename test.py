from procesamiento import DrawingAnalyzer
import json

# Ruta de la imagen de prueba
image_path = "data/temp_prueba1.jpg"  # Asegúrate de que esta imagen exista

# Crear instancia del analizador
analyzer = DrawingAnalyzer()

# Ejecutar análisis
try:
    color_features = analyzer.extract_color_features(image_path)
    shape_features = analyzer.extract_shape_features(image_path)

    print("🎨 Características de color:")
    print(json.dumps(color_features, indent=2, ensure_ascii=False))

    print("\n✏️ Características de forma:")
    print(json.dumps(shape_features, indent=2, ensure_ascii=False))

except Exception as e:
    print("❌ Error al procesar la imagen:", e)
