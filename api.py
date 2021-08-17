from flask import request, jsonify, Flask
import flask
import traceback
import pickle
import pandas as pd

app = Flask(__name__)

# importing models
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)


@app.route('/')
def welcome():
    return "Boston Housing Price Prediction"


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if flask.request.method == 'GET':
        return "Prediction page"
    else:
        try:
            json_ = request.json
            print(json_)
            query_ = pd.get_dummies(pd.DataFrame(json_))
            query = query_.reindex(columns=model_columns, fill_value=0)
            prediction = list(classifier.predict(query))
            return jsonify({"prediction": str(prediction)})
        except:
            return jsonify({"trace": traceback.format_exc()})


if __name__ == "__main__":
    app.run()
