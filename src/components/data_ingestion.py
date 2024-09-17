
import subprocess
from pathlib import Path
import pandas as pd

class DataIngestor:
    def __init__(self, kaggle_source, raw_folder, stg_folder, csv_file):
        self.kaggle_source = kaggle_source
        self.raw_folder = Path(raw_folder)
        self.stg_folder = Path(stg_folder)
        self.csv_file = self.stg_folder / csv_file

    def download_data(self):
        try:
            subprocess.run(f'kaggle datasets download -d {self.kaggle_source}  -p {self.raw_folder} --force', check=True, shell=True)
            print(f"Dataset downloaded successfully at {self.raw_folder}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            return

    def extract_data(self):
        try:
            subprocess.run(f'powershell -command \"Expand-Archive -Force \'{self.raw_folder}/onlineretail.zip\' \'{self.stg_folder}\'\"', check=True, shell=True)
            print(f"Data extracted to {self.stg_folder}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting dataset: {e}")

    def load_data(self):
        # Cargar los datos desde el archivo CSV después de extraerlos
        df = pd.read_csv(self.csv_file, encoding='ISO-8859-1')
        return df

if __name__ == "__main__":
    ingestor = DataIngestor(
        kaggle_source='vijayuv/onlineretail',
        raw_folder='data/raw_data',
        stg_folder='data/stagging_data',
        csv_file='Online Retail.csv'
    )
    ingestor.download_data()
    ingestor.extract_data()
    df = ingestor.load_data()
    print(df.head())
