import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('data/Technical_Support_Dataset.csv')

# Convert datetime columns
df['Created time'] = pd.to_datetime(df['Created time'])
df['Resolution time'] = pd.to_datetime(df['Resolution time'], errors='coerce')
df['Close time'] = pd.to_datetime(df['Close time'], errors='coerce')

# Feature Engineering: Calculate resolution time (in hours)
df['resolution_time_hrs'] = (df['Resolution time'] - df['Created time']).dt.total_seconds() / 3600

# Drop rows where resolution time is missing
df = df.dropna(subset=['resolution_time_hrs'])

# Extract additional time-based features
df['day_of_week'] = df['Created time'].dt.day_name()
df['month'] = df['Created time'].dt.month
df['hour'] = df['Created time'].dt.hour

# Encode categorical features
cat_cols = ['Priority', 'Source', 'Topic', 'Agent Group', 'Agent Name',
            'SLA For first response', 'SLA For Resolution',
            'Product group', 'Support Level', 'Country', 'day_of_week']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Select features and target
features = ['Priority', 'Source', 'Topic', 'Agent Group', 'Agent interactions',
            'Survey results', 'Product group', 'Support Level', 'Country',
            'Latitude', 'Longitude', 'day_of_week', 'month', 'hour']

X = df[features]
y = df['resolution_time_hrs']

# Fill any remaining missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.2f}')
acc=model.score(X_test, y_test)
print(f'Accuracy: {acc:.2f}')
