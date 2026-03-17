import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=200,  
    max_depth=15,      
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained")

# Training accuracy
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_acc*100:.2f}%")

# Testing accuracy
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_acc*100:.2f}%")

print("CONFUSION MATRIX:\n")

cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nCorrect predictions: {cm[0][0] + cm[1][1]} out of {len(y_test)}")
print(f"  - Correctly predicted No Diabetes: {cm[0][0]}")
print(f"  - Correctly predicted Diabetes: {cm[1][1]}")
print(f"\nWrong predictions: {cm[0][1] + cm[1][0]}")
print(f"  - False Positives (predicted diabetes, actually no): {cm[0][1]}")
print(f"  - False Negatives (predicted no diabetes, actually yes): {cm[1][0]}")

print("\n" + "-"*60)
print("DETAILED METRICS")
print("-"*60)
print(classification_report(y_test, y_test_pred, target_names=['No Diabetes', 'Diabetes']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'],yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix (Accuracy: {test_acc*100:.1f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Save model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n" + "="*60)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print("="*60)


