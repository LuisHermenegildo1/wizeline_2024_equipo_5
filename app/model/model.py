import pickle
from pathlib import Path
import re

# Definir las rutas a los artefactos del modelo
model_folder = Path('model')
model_file = model_folder / 'trained_model-1.0.0.pkl'
vectorizer_file = model_folder / 'vectorizer.pkl'

class ModelPredictor:
    def __init__(self):
        # Cargar el modelo entrenado
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        # Cargar el vectorizador (preprocesador)
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def preprocess_text(self, text):
        """
        Preprocesar el texto de entrada: eliminar caracteres innecesarios,
        convertir a minúsculas y normalizar espacios en blanco.
        """
        # Eliminar caracteres especiales y números
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
        
        # Eliminar corchetes correctamente
        text = re.sub(r"[\[\]]", " ", text)
        
        # Convertir a minúsculas
        text = text.lower()

        # Normalizar los espacios en blanco
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Eliminar cualquier punto al final de la cadena (si existe)
        if text.endswith('.'):
            text = text[:-1].strip()

        return text

    def predict(self, description):
        """
        Realiza la predicción del precio basado en la descripción de entrada.
        """
        try:
            # Preprocesar la descripción del producto
            cleaned_text = self.preprocess_text(description)

            # Transformar el texto usando el vectorizador
            input_vector = self.vectorizer.transform([cleaned_text])

            # Realizar la predicción con el modelo cargado
            predicted_price = self.model.predict(input_vector)
            
            # Como es un valor continuo (regresión), devolvemos el precio predicho
            return predicted_price[0]  # Regresar el valor numérico del precio
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")