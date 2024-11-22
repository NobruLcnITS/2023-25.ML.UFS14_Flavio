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


def doit():
    model_dir = environ['SM_MODEL_DIR']
    print(f"######## La model dir è: {model_dir}")
    model = models.load_model(f"{model_dir}/output_model.keras")
    predict_input = np.array([
        [10.19819885, 8.08577004, 0.22707724, 1.0, 0.0, 0.0, 1.0]
    ])
    predict_result = model.predict(predict_input)

    return json.dumps({
        "inputs": predict_input.tolist(),
        "predict_result": predict_result.tolist()
    })
    #return "Funziona"

@app.route('/invocations', methods=['POST'])
def invocations():
    return doit()
    
@app.route('/ping')
def ping():
    return 'Hello, World!'