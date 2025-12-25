import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

n_estimators = 25
max_depth = 5

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy_score = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy",accuracy_score)
    mlflow.log_params("max_depth",max_depth)
    mlflow.log_params("n_estimators",n_estimators)

    print(accuracy_score)
