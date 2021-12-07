import time
from io import StringIO
from flask import Flask, json, request, jsonify
from datetime import datetime as dt
from os import listdir
from os.path import isfile, join, dirname
import pandas as pd
from functions.Prediction import *


app = Flask(__name__)

@app.route('/spike/models')
def getAvailableModels():
    path = join(dirname(__file__), "pickles", "models")
    model_files = [f for f in listdir(path) if isfile(join(path, f))]
    return jsonify(method="getAvailableModels", models=model_files, date=dt.utcnow())

@app.route('/spike/<modelname>/csv', methods=['GET', 'POST'])
def getPredictionFromCSV(modelname:str) -> json:
    if (request.method == "POST"):
        raw = request.get_data().decode()
        _file = StringIO(raw)

        if(raw == ''):
            return jsonify({'error': 'No selected file'})
        else:
            # csv to dataframe
            df = pd.read_csv(_file, index_col=0)

            # predict with model passed by url 
            y_predicted = Predict(df, model_filename=modelname)

            # return result as a json object
            return jsonify(
                model=modelname, 
                prediction=list(y_predicted), 
                date=dt.utcnow())

def main():
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=5000)
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()