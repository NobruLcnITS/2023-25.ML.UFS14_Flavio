import logging
import json
import glob
import sys
from os import environ
from flask import Flask
from keras import models
import numpy as np
from flask import request


logging.debug('Init a Flask app')
app = Flask(__name__)


def doit(lunghezza, diametro):
    model_dir = environ['SM_MODEL_DIR']
    print(f"######## La model dir è: {model_dir}")
    model = models.load_model(f"{model_dir}/output_model.keras")
    predict_input = np.array([
        [57.87785658389723,0.3111400080477545,1.9459399775518593,1.0,1.0,0.0,0.0,0.0]
    ])
    predict_result = model.predict(predict_input)

    return json.dumps({
        "inputs": predict_input.tolist(),
        "predict_result": predict_result.tolist()
    })
    #return "Funziona"

@app.route('/ping')
def ping():
    logging.debug('Hello from route /ping')
    lunghezza = request.args.get('lunghezza')
    diametro = request.args.get('diametro')

    return doit(float(lunghezza), float(diametro))