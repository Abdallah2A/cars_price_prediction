import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

# Load dataset
df = pd.read_csv("../car data.csv").dropna()
df.dropna(inplace=True)

df["Car_Age"] = 2025 - df["Year"]
df.drop(columns="Year", inplace=True)
df.drop(columns="Car_Name", inplace=True)

# Save target variable (Price) and drop unnecessary columns
df.rename(columns={'Selling_Price': 'price_(in_lakhs)'}, inplace=True)
price_col = df.pop("price_(in_lakhs)")

# Encoding categorical features
encoders = {}
categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Save encoders to JSON
with open("../encoders.json", "w") as f:
    json.dump(encoders, f, indent=4, default=str)

# Normalize numerical features
num_cols = ["Car_Age", "Present_Price", "Kms_Driven"]

df[num_cols] = StandardScaler().fit_transform(df[num_cols])

# Save processed data
df.to_csv("../x.csv", index=False)
price_col.to_csv("../y.csv", index=False)

print("✅ Data preprocessing completed. Files saved: x.csv, y.csv, encoders.json")
