## Data ingestion preparation pipeline
###  Data Engineer: Miguel Avila
###  Reviewers: Luis Hermenegildo, Edgar Correa, Diego JE
###  Proyecto Final del BootCamp Wizeline MLOPS2
###  Equipo: mlops-equipo5 Segunda Edicion, 2024

#from https://www.kaggle.com/discussions/general/74235

###dependencies:
#!pip install kaggle

import subprocess

#define config variables
#src_folder = 'data/src_online_retail'
#src_name = 'data/online_retail_raw.csv'
#stg_folder = 'data/stg_online_retail'
#stg_name = 'data/online_retail_stg.csv'
#stg_rejected_name = 'data/online_retail_stg_rejected.csv'
#prod_folder = 'data/prod_online_retail'
#model_train_data = 'data/online_retail_train.csv'
#model_train_pct = 0.6
#model_validation_data = 'data/online_retail_validation.csv'
#model_validation_pct = 0.2
#model_test_data = 'data/online_retail_test.csv'
#model_test_pct = 0.2
#date_format = '%m/%d/%Y %H:%M'

l_kagge_source = 'vijayuv/onlineretail'
l_raw_folder = 'data/raw_data'
l_stg_folder = 'data/stagging_data'

##for windows: key file should be located at c:\<user>\.kaggle
subprocess.run(f'kaggle datasets download -d {l_kagge_source}  -p {l_raw_folder} --force')
subprocess.run(f'powershell -command \"Expand-Archive -Force \'{l_raw_folder}/onlineretail.zip\' \'{l_stg_folder}\'\"')

#! mkdir ~/.kaggle
#! cp data/conf/kaggle.json ~/.kaggle/
#! chmod 600 ~/.kaggle/kaggle.json
#! kaggle datasets list
#kaggle datasets download -d vijayuv/onlineretail
#! mkdir {src_folder}
#! mkdir {stg_folder}
#! mkdir {prod_folder}
#! unzip onlineretail.zip -d {src_folder}/
#! mv {src_folder}/OnlineRetail.csv {src_folder}/{src_name}
