from flask import Flask, render_template, request
import re
import numpy as np
import pandas as pd
from search_embeddings import search_embeddings

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_text():
    user_input = request.form['user_input']
    processed_text = your_processing_function(user_input)
    return render_template('index.html', result=processed_text)

def your_processing_function(text):
    # Implement your processing logic here
    return f'{str(search_embeddings(text))}'

if __name__ == '__main__':
    app.run(debug=True)
