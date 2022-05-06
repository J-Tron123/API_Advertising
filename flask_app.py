import numpy as np, pickle, os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request
from sklearn.linear_model import Lasso

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config["DEBUG"] = True

model = pickle.load(open("ad_model.pkl","rb"))

@app.route("/", methods=["GET"])
def home():
    return "Use '/api/v1/predict' to get your advertising predictions and '/api/v1/retrain' the model with new data"

@app.route("/api/v1/predict", methods=["GET"])
def predict():
    
    tv = request.args.get("tv", None)
    radio = request.args.get("radio", None)
    newspaper = request.args.get("newspaper", None)

    if tv is None or radio is None or newspaper is None:
        return "Args empty, not data to predict"
    else:
        prediction = model.predict([[tv, radio, newspaper]])
    
    return jsonify({"predictions": prediction[0]})


@app.route("/api/v1/retrain", methods=["PUT"])
def retrain():

    data = request.args.get("retraining", None)
    if data is None:
        return "Args empty, not data to retrain"
    else:
        X = data.drop("sales", axis=1)
        y = data["sales"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=24)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        pickle.dump(model, open("data/model/hm_model.pkl", "wb"))

        return jsonify({"Mean Absolute Error" : mean_absolute_error(y_test, predictions), "Root Mean Squared Error" : np.sqrt(mean_squared_error(y_test, predictions))})