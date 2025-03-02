from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "../model/linear_regression_model.pkl")
loaded_model = joblib.load(model_path)


# Define the request model based on dataset features
class PredictionRequest(BaseModel):
    Present_Price: float
    Kms_Driven: float
    Fuel_Type: int
    Seller_Type: int
    Transmission: int
    Owner: int
    Car_Age: float


@app.get("/")
def index():
    return {"message": "Hello World!"}


@app.post("/predict")
def predict(data: list[PredictionRequest]):  # Accepts multiple samples
    # Convert list of objects to a NumPy array
    input_data = np.array([[d.Present_Price, d.Kms_Driven, d.Fuel_Type, d.Seller_Type,
                            d.Transmission, d.Owner, d.Car_Age] for d in data])

    # Make predictions
    predictions = loaded_model.predict(input_data).tolist()  # Convert to list for JSON response

    return {"result": predictions}
