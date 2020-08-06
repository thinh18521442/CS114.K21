# app.py
import random
from flask import Flask, jsonify, request, render_template
from predictor import predict
app = Flask(__name__)

@app.route('/api/calc')
def calc ():
    a = str(request.args.get('data'))
    demo =  predict(a)
    return jsonify({
        "data"        :  demo
    })
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == "__main__":
    app.run()