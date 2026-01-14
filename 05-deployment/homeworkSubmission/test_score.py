import pickle

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


if __name__ == "__main__":
    dv = load_preprocessor(preprocessor_path="./dv.bin")
    model = load_model(model_path="./model1.bin")

    data = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    data_transformed = dv.transform(data)
    pred = model.predict_proba(data_transformed)[:,1]
    print(f"The probability that the client will get the credit card is {round(pred[0],3)}")