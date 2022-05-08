import requests
import json 

url = "http://localhost:9696/predict"

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


data = requests.post(url=url, json=customer)
result = data.json()

print(json.dumps(result, indent=2))

if result["churn"]:
	print("Promotion will be sent")
else:
	print("Customer not likely to churn, hence no need to send promotion")

