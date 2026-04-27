import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("car_data.csv")
df["Car_Age"] = 2026 - df["Year"]
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=12
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("R2 Score:", round(r2_score(y_test, preds), 4))
print("MAE:", round(mean_absolute_error(y_test, preds), 4))

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as car_price_model.pkl")
