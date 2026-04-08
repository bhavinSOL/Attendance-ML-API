import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# 1️⃣ Load CSV data
df = pd.read_csv("attendance.csv")

# 2️⃣ Basic cleaning
df = df.dropna(subset=["absent_percent"])

# 3️⃣ Features & Target
features = [
    "day_of_week",
    "week_number",
    "month",
    "is_holiday",
    "is_festival",
    "festival_weight"
]

X = df[features]
y = df["absent_percent"]

# 4️⃣ Train-Test Split (time-aware)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# 5️⃣ Train Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# 6️⃣ Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("✅ Model trained successfully")
print("📉 Mean Absolute Error:", round(mae, 2))

# 7️⃣ Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 model.pkl saved")
