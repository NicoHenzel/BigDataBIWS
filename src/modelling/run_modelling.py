# Description: This script contains functions for logging the model to the mlflow server.

# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import modelling_functions as mf

# Get the current working directory
cwd = os.getcwd()

# Create the path to the json file
json_path = os.path.join(cwd, 'mlflow', 'gcp', 'bigdatabi-workshop-d2cee8fc15df.json')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path

mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name variable
experiment_name = "Recipe Clustering"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

# If the experiment does not exist, create it
if experiment is None:
    mlflow.create_experiment(experiment_name, artifact_location="gs://artifactbuckethdm/mlartifacts")

mlflow.set_experiment(experiment_name)
mlflow.autolog()

# # Debug params
# file_paths = [
#     "data/processed/df_recipes.pkl"
# ]

# features = [
#     ['preprocessed_ingredients', 'preprocessed_steps', 'preprocessed_description']
# ]

# model_params = {
#     'KMeans': {
#         'model': KMeans(n_init=10, random_state=42),
#         'params': {
#             'n_clusters': [2, 3]
#         }
#     }
# }

file_paths = [
    "data/processed/df_recipes_no_stopwords.pkl",
    "data/processed/df_recipes.pkl"
]

# Prepare features
features = [
    ['preprocessed_ingredients'],
    ['preprocessed_steps'],
    ['preprocessed_description'],
    ['preprocessed_ingredients', 'preprocessed_steps'],
    ['preprocessed_ingredients', 'preprocessed_description'],
    ['preprocessed_steps', 'preprocessed_description'],
    ['preprocessed_ingredients', 'preprocessed_steps', 'preprocessed_description']
]

# Define a list of models
model_params = {
    'KMeans': {
        'model': KMeans(n_init=10, random_state=42),
        'params': {
            'n_clusters': [2, 3, 4, 5]
        }
    },
    'Agglomerative': {
        'model': AgglomerativeClustering(),
        'params': {
            'n_clusters': [2, 3, 4, 5]
        }
    },
    'DBSCAN': {
        'model': DBSCAN(),
        'params': {
            'eps': [0.5, 1.0, 1.5],
            'min_samples': [5, 10, 20]
        }
    }    
}


for file_path in file_paths:
    # Read the data
    df_recipes = pd.read_pickle(file_path)
    # Iterate over features
    for feature in features:
        # Prepare the data for clustering
        df = df_recipes[feature].fillna('')
        
        # Convert any list into string
        for f in feature:
            df[f] = df[f].apply(' '.join)
        
        df['text'] = df[feature].apply(lambda x: ' '.join(x), axis=1)

        # Initialize TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # Fit and transform the vectorizer on the data
        X_sparse = vectorizer.fit_transform(df['text'])

        # Iterate over models
        for model_name, params in model_params.items():
            for p in ParameterGrid(params['params']):
                model = params['model']
                model.set_params(**p)

                # Convert to dense matrix if model is AgglomerativeClustering
                if model_name == 'Agglomerative':
                    X = X_sparse.toarray()
                else:
                    X = X_sparse
                
                # Create a string of the feature names
                feature_combination = '+'.join(feature).replace('preprocessed_', '')

                # Create a run and log the parameters, give unique name to each run, containing the feature combinations    
                with mlflow.start_run(run_name=f'{model_name}_{p}_{feature_combination}'):
                    # Fit the model
                    model.fit(X)
        
                    # Get the cluster predictions
                    labels = model.labels_

                    # Compute Silhouette, Calinski Hrabasz and Davies Bouldin Score only if more than one cluster is found
                    if len(set(labels)) > 1:
                        silhouette_avg = silhouette_score(X, labels)
                        # transform sparse to dense matrix
                        # can be optimized where first transformation takes place
                        if model_name == 'Agglomerative':
                            calinski_harabasz = calinski_harabasz_score(X, labels)
                            davies_bouldin = davies_bouldin_score(X, labels)
                        else:
                            calinski_harabasz = calinski_harabasz_score(X.toarray(), labels)
                            davies_bouldin = davies_bouldin_score(X.toarray(), labels)
                    
                    # write labels to .pkl file for best models
                    mf.save_cluster_labels(df, labels, model_name, p, feature_combination, file_path)

                    # Log Scores
                    mlflow.log_metric("silhouette_score", silhouette_avg)
                    mlflow.log_metric("calinski_harabasz_score", calinski_harabasz)
                    mlflow.log_metric("davies_bouldin_score", davies_bouldin)

                    # Plot clusters
                    mf.plot_and_log_scatterplot(X, labels, model_name, feature)
                    
                    # Plot wordcloud
                    mf.plot_and_log_wordcloud(df, 'text', labels)

                    # Add tags
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("file_path", file_path.removeprefix('data/processed/'))
                    mlflow.set_tag("feature_combination", feature_combination)

