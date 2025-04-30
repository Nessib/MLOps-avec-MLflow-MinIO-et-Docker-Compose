import mlflow
import mlflow.sklearn
import os
import boto3
from sklearn.ensemble import RandomForestClassifier  # Import ajouté
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🧩 Configurer MinIO et les informations d'identification AWS pour MLflow
os.environ['AWS_ACCESS_KEY_ID'] = '3uvPjZjOffW083CyqIHI'  # Remplacer par ta clé d'accès MinIO
os.environ['AWS_SECRET_ACCESS_KEY'] = 'BF20W1tjwpsYAxAtxmVnl77yTzSHE3ej3rTlUvcr'  # Remplacer par ta clé secrète MinIO
# os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'  # Remplacer par la région appropriée (us-east-1 est un exemple)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'  # Remplacer par l'URL de ton serveur MinIO
# 📍 Adresse de ton serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classifier")

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
    mlflow.sklearn.log_model(clf, "model")

    print(f"Run enregistré dans MLflow avec une accuracy de {acc:.2f}")

# 📊 Charger le modèle depuis MinIO
run_id = "fb36306385a3452980ce2363e5b7ccfe"  # Remplacer par le run_id du modèle enregistré
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# 🔍 Faire des prédictions avec le modèle
preds = model.predict(X_test)
print(f"Prédictions : {preds}")