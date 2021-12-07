import pickle5 as pickle
import sys, os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from os.path import join, dirname, split

def TestModel():
    try:
        print("TestingModel")
        # last generated preprocessed data
        data = pickle.load(open(join(dirname(__file__),"..", "pickles", "preprocessed_data.sav"), 'rb'))

        print("dataType:", type(data))
        print(data.head())
        print(data.columns)
        model_1 = pickle.load(open(join(dirname(__file__),"..", "pickles", "models", "model_1"), 'rb'))
        model_2 = pickle.load(open(join(dirname(__file__),"..", "pickles", "models", "model_2"), 'rb'))

        X = data.drop(['Precio_leche'], axis = 1)
        y = data['Precio_leche']

        # generate random data-set
        np.random.seed(0)

        # splitting dataset into 80/20 training/testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("MODEL 1")
        y_predicted = model_1.predict(X_test)
        # evaluar modelo
        rmse = mean_squared_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)
        # printing values
        print('RMSE: ', rmse)
        print('R2: ', r2)

        print("MODEL 2")
        cols_no_leche = [x for x in list(X.columns) if not ('leche' in x)]

        
        X_test = X_test[cols_no_leche]
        y_predicted = model_2.predict(X_test)
        # evaluar modelo
        rmse = mean_squared_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)
        # printing values
        print('RMSE: ', rmse)
        print('R2: ', r2)



    except Exception as e:
        print("Exception - Training()", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pass    


if __name__ == "__main__":
    TestModel()