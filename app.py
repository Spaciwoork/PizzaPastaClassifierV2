#!/usr/bin/python3.7
# coding: utf-8
import random
import torch
from io import BytesIO
from flask import Flask, request, make_response, jsonify

from ModelClassifier import ModelClassifier
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import io


app = Flask(__name__)

ACCESS_TOKEN = 'ACCESS_TOKEN'
VERIFY_TOKEN = 'VERIFY_TOKEN'
cat_to_name = {'0': 'Pizza', '1': 'Pasta'}
model = ModelClassifier()
nn_filename = 'classifier_pizza.pth'
models, optimizer = model.load_checkpoint(nn_filename)

@app.route('/test', methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        image_url = request.args["image"]
        prob, classes = model.predict(urllib.request.urlopen(image_url), models)
        max_index = np.argmax(prob)
        max_probability = prob[max_index]
        val_list = list(cat_to_name.values())
        label = classes[max_index]
        labels = []
        for cl in classes:
            labels.append(cat_to_name[str(val_list.index(cl))])
        message = {
                "messages": [
                    {"text": f'This is a picture of {labels[0]}'}
                ]
            }
        return make_response(jsonify(message))

def launcher():
    use_reloader = True
    app.run(port=5000, host='0.0.0.0', use_reloader=use_reloader)


if __name__ == '__main__':
    launcher()