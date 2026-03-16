
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('diabetes.csv')
print(f"Total rows: {len(df)}")

print(df.head())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"\nColumns: {list(X.columns)}")
print(f"Distribution: No Diabetes={sum(y==0)}, Diabetes={sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train.values)
np.save('y_test.npy', y_test.values)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
