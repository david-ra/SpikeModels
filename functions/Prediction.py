# Predict return Dataframe
from numpy import ndarray
from pandas.core.frame import DataFrame
from os.path import join, dirname, split
import pickle5 as pickle
import sys

def Predict(input_data : DataFrame, model_filename : str) -> ndarray:
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