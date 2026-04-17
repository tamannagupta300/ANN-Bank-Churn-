import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

data = pd.read_csv("Churn_Modelling.csv")

X = data[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']]
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(6,6), activation='logistic', max_iter=500)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
