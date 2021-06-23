import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from preprocessing import *
#from joblib import load
app = Flask(__name__)
model = pickle.load(open('project1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[str(x) for x in request.form.values()]]

    recipe_prediction, recipe_link, imageLink = user_input(x_test)
    print(imageLink)
    return render_template('index.html', recipe_pred = recipe_prediction, recipe_link = recipe_link,ImageLink = imageLink)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)




if __name__ == "__main__":
    app.run(debug=True)
