import pickle
from pathlib import Path
import numpy as np

# Definir las rutas de los archivos del modelo
model_folder = Path('models')
model_file = model_folder / 'trained_model-1.0.0.pkl'
vectorizer_file = model_folder / 'vectorizer.pkl'
decoder_file = model_folder / 'language_dec-1.0.0.pkl'

class PredictPipeline:
    def __init__(self):
        # Cargar el modelo entrenado
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Cargar el vectorizador (preprocesador)
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Cargar el decodificador
        with open(decoder_file, 'rb') as f:
            self.decoder = pickle.load(f)

    def predict(self, description):
        """
        Realiza la predicción basado en la descripción de entrada.
        """
        try:
            # Transformar la entrada usando el vectorizador
            input_vector = self.vectorizer.transform([description])
            
            # Realizar la predicción con el modelo cargado
            prediction = self.model.predict(input_vector)
            
            # Decodificar la predicción a su valor de etiqueta original (país)
            predicted_country = self.decoder[prediction[0]]
            
            return predicted_country
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
