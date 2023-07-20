@echo off
set GOOGLE_APPLICATION_CREDENTIALS=%~dp0gcp\bigdatabi-workshop-d2cee8fc15df.json
echo %GOOGLE_APPLICATION_CREDENTIALS%
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root gs://artifactbuckethdm/mlartifacts --host 0.0.0.0