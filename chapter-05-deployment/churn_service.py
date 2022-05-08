from flask import Flask, request, jsonify
from predict import predict_single

app = Flask('ping')


@app.route(rule="/test", methods=["GET"])
def test():
    return 'SERVICE IS UP'


@app.route(rule="/predict", methods=["POST"])
def run_predict():

    data = request.get_json()
    prediction = predict_single(data)

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='9696')
