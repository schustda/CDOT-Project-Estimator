from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.externals import joblib
from src.model.model import CDOTModel
import locale

app = Flask(__name__)
locale.setlocale(locale.LC_ALL,'')

@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    feature_url = 'https://docs.google.com/spreadsheets/d/1Zt-KDYQP80NmwiWyX0hzukuyBO5YExwohWA9tBqGPhw/export?format=csv&id=1Zt-KDYQP80NmwiWyX0hzukuyBO5YExwohWA9tBqGPhw'
    X_predict = pd.read_csv(feature_url,usecols = ['Quantity']).values.reshape(1,-1)
    data = str(request.form['Item_1'])
    model = CDOTModel()
    pred = locale.currency(model.predict(X_predict),grouping = True)
    return render_template('predict.html', predicted = pred)
    # return render_template('predict.html', article=data, predicted = pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, threaded=True)
