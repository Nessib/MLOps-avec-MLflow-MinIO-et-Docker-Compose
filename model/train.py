import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Variables d'environnement pour MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # URL de ton MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minio"  # Utilisateur MinIO
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"  # Mot de passe MinIO

# 📍 Adresse de votre serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Pointage vers ton serveur MLflow
mlflow.set_experiment("iris-classifier")  # Expérience

# 🔄 Chargement des données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# 🧪 Expérience MLflow
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, max_depth=3)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # 📝 Log paramètres et métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc)

    # 💾 Log du modèle dans MinIO via MLflow
    mlflow.sklearn.log_model(clf, "model")  # Enregistre le modèle dans MinIO

    print(f"Run enregistré dans MLflow avec une accuracy de {acc:.2f}")
