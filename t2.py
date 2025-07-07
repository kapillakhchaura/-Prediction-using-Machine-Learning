# ✅ Step 1: Required Libraries Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Step 2: Define column names (from heart-disease.names file)
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'  # 'target' == 'num' in original dataset
]

# ✅ Step 3: Load dataset
df = pd.read_csv('processed.cleveland.data', names=column_names)

# ✅ Step 4: Handle missing values represented by '?'
df.replace('?', pd.NA, inplace=True)  # Replace ? with NaN
df.dropna(inplace=True)               # Drop rows with missing values

# ✅ Step 5: Convert all columns to numeric type
df = df.apply(pd.to_numeric)

# ✅ Step 6: Split Features and Target
X = df.drop('target', axis=1)  # Input features
y = df['target']               # Target column

# ✅ Step 7: Convert target to binary (0 = No disease, 1+ = Disease)
y = y.apply(lambda val: 1 if val > 0 else 0)

# ✅ Step 8: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Step 9: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ✅ Step 10: Predict and Evaluate
y_pred = model.predict(X_test)

# ✅ Step 11: Print Results
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
