from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.model import ModelPredictor

# Inicializar FastAPI
app = FastAPI()

# Inicializar el predictor del modelo para la predicción de precios
model_predictor = ModelPredictor()

# Clase para la entrada
class PredictionInput(BaseModel):
    description: str

# Clase para la salida
class PredictionOutput(BaseModel):
    predicted_price: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1.0.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    try:
        # Llamar al predictor para hacer una predicción de precio
        predicted_price = model_predictor.predict(payload.description)
        return {"predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")