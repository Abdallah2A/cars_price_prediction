import requests

url = "http://127.0.0.1:8000/predict"

# Sample input
data = [
    {
        "Present_Price": -0.236214614,
        "Kms_Driven": -0.256224461,
        "Fuel_Type": 2,
        "Seller_Type": 0,
        "Transmission": 1,
        "Owner": 0,
        "Car_Age": -0.128897003
    },
    {
        "Present_Price": 0.221504617,
        "Kms_Driven": 0.155910503,
        "Fuel_Type": 1,
        "Seller_Type": 0,
        "Transmission": 1,
        "Owner": 0,
        "Car_Age": 0.217513693
    }
]

response = requests.post(url, json=data)
print(response.json())  # Output: {"result": [prediction1, prediction2]}
