🧪 MLOps Project with MLflow, MinIO, and Docker Compose
This project provides a local MLOps environment for tracking ML experiments using MLflow, with MinIO as the artifact backend, all orchestrated using Docker Compose.

🚀 Getting Started
Clone the repository:
bash
Copier
Modifier
git clone <repo-url>
cd <repo-name>
Start the services with Docker Compose:
bash
Copier
Modifier
docker-compose up --build
Create the MinIO bucket (if not automated):
bash
Copier
Modifier
./create_minio_bucket.sh
🌐 Access the Services
MLflow UI: http://localhost:5000

MinIO Console: http://localhost:9001
