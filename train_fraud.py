import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

print("1. Loading PaySim dataset...")
# Load 300,000 rows to get more fraud examples
df = pd.read_csv("paysim.csv", nrows=300000)

print("2. Filtering data...")
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
df['type'] = df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})

features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
target = 'isFraud'

X = df[features]
y = df[target]

print("3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Applying SMOTE to balance fraud data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("5. Training model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

print("\n6. Model Evaluation:")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

joblib.dump(model, "fraud_model.pkl")
print("\nDone! Model saved as fraud_model.pkl")