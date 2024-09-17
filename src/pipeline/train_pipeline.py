from src.components.data_ingestion import DataIngestor
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    # 1. Ingesta de datos
    print("Starting data ingestion...")
    ingestor = DataIngestor(
        kaggle_source='vijayuv/onlineretail',
        raw_folder='data/raw_data',
        stg_folder='data/stagging_data',
        csv_file='OnlineRetail.csv'
    )
    ingestor.download_data()
    ingestor.extract_data()
    df = ingestor.load_data()

    if df is None:
        print("Error: No data found, terminating pipeline.")
        return

    # 2. Transformación de datos
    print("Starting data transformation...")
    transformer = DataTransformer(
        stg_folder='data/stagging_data',
        prod_folder='data/production_data',
        prod_file='OnlineRetail_clean.csv'
    )
    df_clean = transformer.clean_data(df)
    X_transformed = transformer.fit_transform(df_clean)
    y = df_clean['Quantity']  # Usando 'Quantity' como la columna objetivo para la predicción

    # Guardar los datos transformados
    transformer.save_clean_data(df_clean)

    # 3. Entrenamiento de modelos
    print("Starting model training...")
    trainer = ModelTrainer()

    # Entrenamiento usando GridSearchCV
    best_models, model_scores = trainer.grid_search(X_transformed, y)

    # Aquí podríamos realizar un entrenamiento con RandomizedSearch si quieres comparar
    # best_models_random, model_scores_random = trainer.random_search(X_transformed, y)

    # 4. Guardar los modelos entrenados y otros artefactos necesarios
    # Nota: Aquí podríamos guardar solo el mejor modelo o múltiples modelos si es necesario
    for model_name, model in best_models.items():
        print(f"Saving {model_name}...")
        trainer.save_model(model, transformer.preprocessor, y)

    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    run_training_pipeline()
