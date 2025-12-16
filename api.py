import os, re, cv2, json, csv, uvicorn
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from procesamiento import DrawingAnalyzer
from emociones import EmotionClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Emociones IA API", version="1.0.0")

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

drawing_analyzer = DrawingAnalyzer()
emotion_classifier = EmotionClassifier()

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    emotional_analysis: dict
    color_analysis: dict
    shape_analysis: dict
    recommendations: list
    file_url: str

@app.post("/analyze-drawing", response_model=AnalysisResponse)
async def analyze_drawing(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        file_url = f"http://127.0.0.1:8000/uploads/{safe_filename}"

        color_analysis = drawing_analyzer.extract_color_features(file_path)
        shape_analysis = drawing_analyzer.extract_shape_features(file_path)
        dominant_colors = drawing_analyzer.extract_dominant_colors(file_path)
        color_analysis["dominant_colors"] = dominant_colors

        with open(file_path, "rb") as f:
            image_bytes = f.read()
            emotional_analysis = emotion_classifier.predict_emotion(image_bytes)

        recommendations = generate_recommendations(emotional_analysis, color_analysis)

        response = AnalysisResponse(
            analysis_id=generate_analysis_id(),
            timestamp=datetime.now().isoformat(),
            emotional_analysis=emotional_analysis,
            color_analysis=color_analysis,
            shape_analysis=shape_analysis,
            recommendations=recommendations,
            file_url=file_url
        )

        guardar_resultado_json(response.dict())
        guardar_resultado_csv(response.dict())

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el análisis: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def generate_recommendations(emotional_analysis, color_analysis):
    emotion = emotional_analysis.get("emotion")
    confidence = emotional_analysis.get("confidence", 0)
    recommendations = []
    if emotion == "tristeza" and confidence > 0.7:
        recommendations.append("Fomentar actividades con colores cálidos y ejercicios creativos")
    elif emotion == "enojo" and confidence > 0.6:
        recommendations.append("Técnicas de relajación y trazos suaves recomendados")
    return recommendations

def generate_analysis_id():
    return f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def guardar_resultado_json(resultado: dict, nombre_archivo: str = "resultados.json"):
    archivo = os.path.join(BASE_DIR, nombre_archivo)
    with open(archivo, "a", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False)
        f.write("\n")

def guardar_resultado_csv(resultado: dict, nombre_archivo: str = "resultados.csv"):
    archivo = os.path.join(BASE_DIR, nombre_archivo)
    campos = ["timestamp", "color_analysis", "shape_analysis", "emotional_analysis"]
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "timestamp": resultado.get("timestamp"),
            "color_analysis": json.dumps(resultado.get("color_analysis", ""), ensure_ascii=False),
            "shape_analysis": json.dumps(resultado.get("shape_analysis", ""), ensure_ascii=False),
            "emotional_analysis": json.dumps(resultado.get("emotional_analysis", ""), ensure_ascii=False),
        })

# if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)