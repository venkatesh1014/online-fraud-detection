import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv(r'C:\Users\malap\OneDrive\Desktop\venky_project\data\PS_20174392719_1491204439457_log.csv')

# 2. Preprocessing
df.drop(['isFlaggedFraud'], axis=1, inplace=True)

# Sampling 200,000 rows for speed (Balanced enough for training)
df = df.sample(n=200000, random_state=42)

# Handling Outliers
q1, q3 = df['amount'].quantile(0.25), df['amount'].quantile(0.75)
IQR = q3 - q1
df = df[(df['amount'] >= q1 - 1.5*IQR) & (df['amount'] <= q3 + 1.5*IQR)]

# Encoding
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Features and Target
X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
y = df['isFraud']

# 3. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling is CRITICAL for SVM performance and speed
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Training (using LinearSVC for speed on large datasets)
print("Training model...")
model = LinearSVC(max_iter=5000)
model.fit(X_train, y_train)

# 5. Save Model AND Scaler
# We must save the scaler to use it on user inputs in the web app
with open('payments.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
print("Model and Scaler saved successfully!")