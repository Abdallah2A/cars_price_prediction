# Cars Price Prediction

## Overview
This project predicts the selling price of used cars based on various features using a Linear Regression model. 
The dataset is preprocessed, trained, and then deployed as an API using FastAPI, which is containerized with Docker.

## Project Structure
```
cars_price_prediction/
│── api/
│   │── app.py             # FastAPI app with prediction endpoint
│   │── main.py            # Entry point for API (**runs the FastAPI app using Uvicorn**)
│   │── Dockerfile         # Docker configuration
│   │── requirements.txt   # Dependencies
│── data/
│   │── scripts/
│   │   │── car data.csv   # Raw dataset
│   │   │── encoders.json  # Encoded categorical values
│   │   │── x.csv          # Processed input features
│   │   │── y.csv          # Target variable
│── model/
│   │── training/
│   │   │── model.py       # train model using sklearn
│   │   │── scratch.py     # train model from scratch
│   │── custom_linear_regression.pkl  # Custom-trained model
│   │── linear_regression_model.pkl       # Final trained model
│── tests/
│   │── test_api.py        # API testing script
│── requirements.txt   # Testing dependencies
```

## Setup Instructions

### **1. Install Dependencies**
Make sure you have Python 3.12 installed.
```bash
pip install -r requirements.txt
```

### **2. Data Preprocessing**
Run the preprocessing script to clean and encode the dataset.
```bash
python data/scripts/preprocess.py
```

### **3. Model Training**
Train the Linear Regression model and save it.
```bash
python model/training/model.py
```

### **4. Run the API Locally**
Start the FastAPI server.
```bash
python api/main.py
```

### **5. Docker Deployment**
Build and run the Docker container.
```bash
docker build -t cars-price-api ./api
docker run -p 8000:8000 cars-price-api
```

## **Testing**
Run API test:
```bash
pytest tests/test_api.py
```

## **License**
This project is open-source and free to use.