#!/bin/bash

# Attendre que MinIO soit disponible
echo "Waiting for MinIO to be ready..."
sleep 10

# Ajouter MinIO à mc (MinIO Client)
mc alias set myminio http://minio:9000 minio minio123

# Créer le bucket mlflow-artifacts si non existant
mc mb myminio/mlflow-artifacts || echo "Bucket mlflow-artifacts already exists"

# Afficher les buckets disponibles
mc ls myminio
