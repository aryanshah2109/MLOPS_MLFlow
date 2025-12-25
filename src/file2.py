import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

n_estimators = 12
max_depth = 3


dagshub.init(repo_owner='aryanshah2109', repo_name='MLOPS_MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/aryanshah2109/MLOPS_MLFlow.mlflow/")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_params({
        "max_depth":max_depth
    })
    mlflow.log_params({
        "n_estimators":n_estimators
    })

    print(accuracy)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.savefig("Confusion-Matrix.png")

    mlflow.log_artifact("Confusion-Matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({
        "Author" : "Aryan",
        "Project_Name": "Wine Classification"
    })

    mlflow.sklearn.log_model(model, "Random-Forest-Model")