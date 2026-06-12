import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the CSV dataset
# (File name-a unga CSV file-kku ethapola mathikonga)
df = pd.read_csv('student_performance_data.csv')

# 2. Define Features (X) and Target (y)
# Make sure 'student_id' and 'performance_level' match your CSV exact column names
X = df.drop(columns=['student_id', 'performance_level'])
y = df['performance_level']

# 3. Convert categorical text columns into numeric indicator variables
X = pd.get_dummies(X, drop_first=True)

# Save the exact column names so our web form aligns perfectly
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# 4. Train the model using all features
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. Save the trained model
joblib.dump(model, 'model.pkl')

print("Model trained using ALL features and saved successfully as model.pkl")