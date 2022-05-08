import pickle


model_location = "model.pkl"


def predict(data):
    """Function that does batch prediction"""


    with open(model_location, 'rb') as file:
        dv, model = pickle.load(file)

    
    X = dv.transform(data)
    churn_prob = model.predict_proba(X)[:,1]
    churn = (churn_prob >= 0.5)

    prediction = {"probability_of_churn": churn_prob, "churn": churn}

    return prediction

def predict_single(data):
    """Function that predicts single value"""

    
    with open(model_location, 'rb') as file:
        dv, model = pickle.load(file)

    
    X = dv.transform(data)
    churn_prob = model.predict_proba(X)[:,1]
    churn = (churn_prob >= 0.5)

    prediction = {"probability_of_churn": churn_prob[0], "churn": bool(churn[0])}

    return prediction



if __name__ == "__main__":

    customer = {'tenure': 1, 
            'monthlycharges': 29.85, 
            'totalcharges': 29.85, 
            'gender': 'female', 
            'seniorcitizen': 0, 
            'partner': 'yes', 
            'dependents': 'no', 
            'phoneservice': 'no', 
            'multiplelines': 'no_phone_service',
            'internetservice': 'dsl', 
            'onlinesecurity': 'no', 
            'onlinebackup': 'yes', 
            'deviceprotection': 'no', 
            'techsupport': 'no', 
            'streamingtv': 'no', 
            'streamingmovies': 'no', 
            'contract': 'month-to-month', 
            'paperlessbilling': 'yes', 
            'paymentmethod': 'electronic_check'}

    print(predict(customer))
