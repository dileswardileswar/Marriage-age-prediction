import flask
from flask import Flask, request, jsonify, render_template
app = flask.Flask(__name__)
app.config["DEBUG"] = True


# main index page route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    from sklearn.externals import joblib
    model = joblib.load('marriage_age_predict_model.ml')
    predicted_age_of_marriage = model.predict([[int(request.args['gender']),
                            int(request.args['religion']),
                            int(request.args['caste']),
                            int(request.args['mother_tongue']),
                            int(request.args['country']),
                            int(request.args['height_cms']),
                           ]])
    return str(round(predicted_age_of_marriage[0],2))


if __name__ == "__main__":
    app.run()
