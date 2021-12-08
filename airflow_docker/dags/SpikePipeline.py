
from airflow.decorators import dag
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator, PythonVirtualenvOperator
from os.path import dirname, join
from datetime import datetime as dt
import logging

default_args = {
    'owner': 'spike',
    #'retries': 1,
    'csv_folder': "~/data",
    'pickles_folder': "~/pickles",
    'start_date': dt(2021,1,1)
}

# define the DAG
dag = DAG(
    'SpikeModelGenerator',
    default_args=default_args,
    description='Training Models 1/2 for ML Spike challenge',
    # schedule_interval=timedelta(days=1),
)

def preProcessingData(path_file_1, path_file_2, path_file_3):
    import pandas as pd
    #import locale
    #locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    #locale.setlocale(locale.LC_ALL,'es_ES.UTF-8')
    import logging
    import pickle5 as pickle
    from os.path import split
    import sys

    def convert_int(x):
        return int(x.replace('.', ''))

    def to_100(x): #mirando datos del bc, pib existe entre ~85-120 - igual esto es cm (?)
        x = x.split('.')
        if x[0].startswith('1'): #es 100+
            if len(x[0]) >2:
                return float(x[0] + '.' + x[1])
            else:
                x = x[0]+x[1]
                return float(x[0:3] + '.' + x[3:])
        else:
            if len(x[0])>2:
                return float(x[0][0:2] + '.' + x[0][-1])
            else:
                x = x[0] + x[1]
                return float(x[0:2] + '.' + x[2:])


    try:
        esp_months = {'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12}
        # CLEANING DATA =======================================================
        # reading files 
        #precipitaciones = pd.read_csv('./data/precipitaciones.csv')
        precipitaciones = pd.read_csv(path_file_1)
        precipitaciones['date'] = pd.to_datetime(precipitaciones['date'], format = '%Y-%m-%d')
        precipitaciones = precipitaciones.sort_values(by = 'date', ascending = True).reset_index(drop = True)
        
        # deleting nan values
        precipitaciones[precipitaciones.isna().any(axis=1)] 
        # deleting repeated values
        precipitaciones[precipitaciones.duplicated(subset = 'date', keep = False)]

        regiones = ['Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago',
        'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio',
        'La_Araucania', 'Los_Rios']

        precipitaciones[regiones].describe() 

        # reading central bank variables
        #banco_central = pd.read_csv('../data/banco_central.csv')
        banco_central = pd.read_csv(path_file_2)
        banco_central['Periodo'] = banco_central['Periodo'].apply(lambda x: x[0:10])
        banco_central['Periodo'] = pd.to_datetime(banco_central['Periodo'], format = '%Y-%m-%d', errors = 'coerce')
        #print(banco_central.columns) 

        # deleting repeated values
        banco_central[banco_central.duplicated(subset = 'Periodo', keep = False)]
        banco_central.drop_duplicates(subset = 'Periodo', inplace = True)

        # deleting NaN values
        banco_central = banco_central[~banco_central.Periodo.isna()]



        cols_pib = [x for x in list(banco_central.columns) if 'PIB' in x]
        cols_pib.extend(['Periodo'])
        banco_central_pib = banco_central[cols_pib]
        banco_central_pib = banco_central_pib.dropna(how = 'any', axis = 0)

        for col in cols_pib:
            if col == 'Periodo':
                continue
            else:
                banco_central_pib[col] = banco_central_pib[col].apply(lambda x: int(x.replace('.', '')))

        banco_central_pib.sort_values(by = 'Periodo', ascending = True)
        #banco_central_pib

        cols_imacec = [x for x in list(banco_central.columns) if 'Imacec' in x]
        cols_imacec.extend(['Periodo'])
        banco_central_imacec = banco_central[cols_imacec]
        banco_central_imacec = banco_central_imacec.dropna(how = 'any', axis = 0)

        for col in cols_imacec:
            if col == 'Periodo':
                continue
            else:
                banco_central_imacec[col] = banco_central_imacec[col].apply(lambda x: to_100(x))
                assert(banco_central_imacec[col].max()>100)
                assert(banco_central_imacec[col].min()>30)

        banco_central_imacec.sort_values(by = 'Periodo', ascending = True)

        banco_central_iv = banco_central[['Indice_de_ventas_comercio_real_no_durables_IVCM', 'Periodo']]
        banco_central_iv = banco_central_iv.dropna() # -unidades? #parte 
        banco_central_iv = banco_central_iv.sort_values(by = 'Periodo', ascending = True)

        banco_central_iv['num'] = banco_central_iv.Indice_de_ventas_comercio_real_no_durables_IVCM.apply(lambda x: to_100(x))

        banco_central_num = pd.merge(banco_central_pib, banco_central_imacec, on = 'Periodo', how = 'inner')
        banco_central_num = pd.merge(banco_central_num, banco_central_iv, on = 'Periodo', how = 'inner')


        # ==========================================================================================
        # ==========================================================================================
        #precio_leche = pd.read_csv('../data/precio_leche.csv')
        precio_leche = pd.read_csv(path_file_3)
        precio_leche.rename(columns = {'Anio': 'ano', 'Mes': 'mes_pal'}, inplace = True) # precio = nominal, sin iva en clp/litro
        
        print("mes_pal:", precio_leche['mes_pal'])
        
        #precio_leche['mes'] = pd.to_datetime(precio_leche['mes_pal'], format = '%b')
        precio_leche['mes'] = precio_leche['mes_pal'].map(esp_months)

        #precio_leche['mes'] = precio_leche['mes'].apply(lambda x: x.month)
        precio_leche['mes-ano'] = precio_leche.apply(lambda x: f'{x.mes}-{x.ano}', axis = 1)

        precipitaciones['mes'] = precipitaciones.date.apply(lambda x: x.month)
        precipitaciones['ano'] = precipitaciones.date.apply(lambda x: x.year)
        precio_leche_pp = pd.merge(precio_leche, precipitaciones, on = ['mes', 'ano'], how = 'inner')
        precio_leche_pp.drop('date', axis = 1, inplace = True)


        banco_central_num['mes'] = banco_central_num['Periodo'].apply(lambda x: x.month)
        banco_central_num['ano'] = banco_central_num['Periodo'].apply(lambda x: x.year)
        precio_leche_pp_pib = pd.merge(precio_leche_pp, banco_central_num, on = ['mes', 'ano'], how = 'inner')
        precio_leche_pp_pib.drop(['Periodo', 'Indice_de_ventas_comercio_real_no_durables_IVCM', 'mes-ano', 'mes_pal'], axis =1, inplace = True)

        cc_cols = [x for x in precio_leche_pp_pib.columns if x not in ['ano', 'mes']]

        precio_leche_pp_pib_shift3_mean = precio_leche_pp_pib[cc_cols].rolling(window=3, min_periods=1).mean().shift(1)
        precio_leche_pp_pib_shift3_mean.columns = [x+'_shift3_mean' for x in precio_leche_pp_pib_shift3_mean.columns]                                        
        precio_leche_pp_pib_shift3_std = precio_leche_pp_pib[cc_cols].rolling(window=3, min_periods=1).std().shift(1)
        precio_leche_pp_pib_shift3_std.columns = [x+'_shift3_std' for x in precio_leche_pp_pib_shift3_std.columns] 
        precio_leche_pp_pib_shift1 = precio_leche_pp_pib[cc_cols].shift(1)
        precio_leche_pp_pib_shift1.columns = [x+'_mes_anterior' for x in precio_leche_pp_pib_shift1.columns]

        precio_leche_pp_pib = pd.concat([precio_leche_pp_pib['Precio_leche'], precio_leche_pp_pib_shift3_mean, precio_leche_pp_pib_shift3_std, precio_leche_pp_pib_shift1], axis = 1) 
        precio_leche_pp_pib = precio_leche_pp_pib.dropna(how = 'any', axis = 0)
        #precio_leche_pp_pib.head()

        # store file in picklefile folder
        pickle.dump(precio_leche_pp_pib, open("/opt/airflow/pickles/preprocessed_data.sav", 'wb'))

        #return precio_leche_pp_pib

    except Exception as e:
        print("Exception - preProcessingData()", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        logging.error(exc_type, fname, exc_tb.tb_lineno)

# Training Stage with default values 
def Training(datapath, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    import numpy as np
    from os.path import split
    import sys
    import pickle5 as pickle

    try:
        data = pickle.load(open(datapath, 'rb'))

        print("dataType:", type(data))
        print("Training Stage...")
        print("MODEL_1")

        # MODEL VARIABLES
        X = data.drop(['Precio_leche'], axis = 1)
        y = data['Precio_leche']

        print(X.head(),y.head())

        # generate random data-set
        np.random.seed(0)

        # splitting dataset into 80/20 training/testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        pipe = Pipeline([('scale', StandardScaler()),
                    ('selector', SelectKBest(mutual_info_regression)),
                    ('poly', PolynomialFeatures()),
                    ('model', Ridge())])
        
        k=[3, 4, 5, 6, 7, 10]
        alpha=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        poly = [1, 2, 3, 5, 7]
        model_1 = GridSearchCV(
            estimator = pipe,
            param_grid = dict(
                selector__k=k, 
                poly__degree=poly, 
                model__alpha=alpha
            ), 
            cv = 3, 
            scoring = 'r2')

        # fitting grid model_1
        model_1.fit(X_train, y_train)


        # generate random data-set
        print("MODEL_2")
        np.random.seed(0)
        cols_no_leche = [x for x in list(X.columns) if not ('leche' in x)]
        X_train = X_train[cols_no_leche]
        X_test = X_test[cols_no_leche]

        pipe = Pipeline([('scale', StandardScaler()),
                        ('selector', SelectKBest(mutual_info_regression)),
                        ('poly', PolynomialFeatures()),
                        ('model', Ridge())])
        k=[3, 4, 5, 6, 7, 10]
        alpha=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        poly = [1, 2, 3, 5, 7]
        model_2 = GridSearchCV(estimator = pipe,
                            param_grid = dict(selector__k=k,
                                            poly__degree=poly,
                                            model__alpha=alpha),
                            cv = 3,
                        scoring = 'r2')

        model_2.fit(X_train, y_train)

        pickle.dump(model_1, open("/opt/airflow/pickles/models/model_1", 'wb'))
        pickle.dump(model_2, open("/opt/airflow/pickles/models/model_2", 'wb'))

        #return model_1, model_2

    except Exception as e:
        print("Exception - Training()", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def Predict(input_data, model_filename : str):
    import pickle5 as pickle
    try:
        print(f"Predict() using {model_filename}")
        
        y_predicted =  None
        model = pickle.load(open(join(dirname(__file__), "..", "pickles", "models", model_filename), 'rb'))

        if(model):
            y_predicted = model.predict(input_data)   
        else:
            print(f"Model not found")

        return y_predicted

    except Exception as e:
        print("Exception - Predict()", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass  


pre_process = PythonVirtualenvOperator(
    task_id='pre-process',
    python_callable=preProcessingData,
    requirements=["pandas", "sklearn", "pickle5"],
    op_kwargs={
        "path_file_1":"/opt/airflow/data/precipitaciones.csv",
        "path_file_2":"/opt/airflow/data/banco_central.csv",
        "path_file_3":"/opt/airflow/data/precio_leche.csv"
    },
    dag=dag
)

import pickle
training = PythonVirtualenvOperator(
    task_id='Training',
    python_callable=Training,
    requirements=["pandas", "sklearn", "pickle5"],
    op_kwargs={
        "datapath": "/opt/airflow/pickles/preprocessed_data.sav"
    },
    dag=dag
)

pre_process >> training


