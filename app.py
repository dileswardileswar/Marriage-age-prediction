import flask
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


app = flask.Flask(__name__)
model = pickle.load(open('marriage_age_predict_model.pkl','rb'))
app.config["DEBUG"] = True


# main index page route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():    
    predicted_age_of_marriage = [int(value) for value in request.form.values()]
    final_predicted_age_of_marriage = [np.array(predicted_age_of_marriage)]
    prediction = model.predict(final_predicted_age_of_marriage)
    output = int(prediction[0])
    return render_template('index.html', prediction_text = "Your Predicted Year of Marriage : {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
