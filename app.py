import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model1 = pickle.load(open('logreg','rb'))
model2 = pickle.load(open('Randf','rb'))
model3 = pickle.load(open('nb','rb'))

vect = CountVectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    try:
        text = flask.request.form.get('text')
        prediction1 = model1.predict(vect.transform([text]))[0]
        prediction2 = model2.predict(vect.transform([text]))[0]
        prediction3 = model3.predict(vect.transform([text]))[0]
    except:
        print("Block of code has an error")
        
    output1= prediction1
    output2 = prediction2
    output3 = prediction3
    
    return render_template('index.html', pred_text1 = 'LogiticRegression = {}'.format(output1))
    return render_template('index.html', pred_text2 = 'Randf = {}'.format(output2))
    return render_template('index.html', pred_text3 = 'Naive Baise = {}'.format(output3))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
