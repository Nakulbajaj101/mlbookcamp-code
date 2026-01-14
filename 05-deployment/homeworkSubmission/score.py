import pickle

from flask import Flask, jsonify, request


app = Flask("CreditCardService")

def load_pickle_obj(path):
    """Function to load pickle object"""

    with open(path, 'rb') as file:
        obj = pickle.loads(file.read())
    return obj

def load_preprocessor(preprocessor_path):
    """Function to load transformer object"""

    preprocessor = load_pickle_obj(path=preprocessor_path)
    return preprocessor

def load_model(model_path):
    """Function to load model object"""

    model = load_pickle_obj(path=model_path)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    """Predicting the outcome"""

    dv = load_preprocessor(preprocessor_path="./dv.bin")
    model = load_model(model_path="./model2.bin")
    data = request.get_json()
    data_transformed = dv.transform(data)
    pred = model.predict_proba(data_transformed)[:,1][0]

    credi_card = pred >= 0.5

    result = {
        'credit_card_prob': round(pred,3),
        'credit_card_approve': bool(credi_card)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
