import bentoml
from bentoml.io import JSON
from pydantic import BaseModel


class CreditServiceData(BaseModel):
    
    seniority : int
    home : str
    time : int
    age : int
    marital : str
    records : str
    job : str
    expenses : int
    income : float
    assets : float
    debt : float
    amount : int
    price : int



model_ref = bentoml.xgboost.get(
    tag_like="credit_risk_model:latest"
)

dv = model_ref.custom_objects["preprocessor"]

runner = model_ref.to_runner()


svc = bentoml.Service(
    name="credit_risk_classifier", runners=[runner]
)

@svc.api(input=JSON(pydantic_model=CreditServiceData), output=JSON())
async def classify(raw_data):

    app_data = raw_data.dict()
    vector = dv.transform(app_data)
    prediction = await runner.predict.async_run(vector)

    result = prediction[0]

    if result > 0.5:
        return {
            "status": "Declined"
            }
    elif result > 0.25:
        return {
            "status": "Declined"
        }
    else: 
        return {
            "status": "Approved"
            }
