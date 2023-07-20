import io
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import mlflow
from wordcloud import WordCloud

def plot_and_log_scatterplot(X, labels, model_name, feature):
    """
    Plots a 2D scatter plot of the reduced features and saves it to a file.
    The file is then logged to MLflow.

    Parameters
    ----------
    X : array-like
        The features to plot.
    labels : array-like of shape (n_samples,)
        The cluster labels.
    model_name : str
        The name of the model.
    params : dict
        The parameters of the model.
    feature : list containing strings of features.
    """
    # Create SVD model and fit it to the data
    svd = TruncatedSVD(n_components=2)
    reduced_features = svd.fit_transform(X)

    # Create a string of the feature names
    feature_combination = '+'.join(feature).replace('preprocessed_', '')

    # Plot the figure
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title(f'Visualization of {model_name} Clusters containing {feature_combination}')
    plt.colorbar()

    # Create a temporary directory using the context manager
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define the filename
        filename = os.path.join(tmpdirname, "clusters.png")
        
        # Save the plot to a file within the temporary directory
        plt.savefig(filename, format='png')

        # Log plot
        mlflow.log_artifact(filename)



def plot_and_log_wordcloud(df, column, labels):
    """
    Plots a word cloud for each unique cluster in the given data and saves them to files.
    The files are then logged to MLflow.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the text data.
    column : str
        The name of the column in df that contains the text data.
    labels : array-like
        The cluster labels for each data point.
    """
    # Create a dictionary mapping each unique label to the corresponding text
    cluster_to_text = {label: ' '.join(df[column][labels == label]) for label in set(labels)}

    for label, text in cluster_to_text.items():

         # Skip if there are no words for this cluster
        if not text.strip():  # Check if text is not empty after removing leading/trailing whitespaces
            continue

        try:
            # Generate the word cloud
            wordcloud = WordCloud(max_font_size=100).generate(text)

            # Plot the word cloud
            plt.figure()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")

            # Create a temporary directory using the context manager
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Define the filename
                filename = os.path.join(tmpdirname, f"wordcloud_{label}.png")
                
                # Save the plot to a file within the temporary directory
                plt.savefig(filename, format='png')

                # Log plot
                mlflow.log_artifact(filename)

        except ValueError:
            continue

        # Close the plot to free memory
        finally:
            plt.close()


def save_cluster_labels(df, labels, model_name, p, feature_combination, file_path):
    """
    Saves the cluster labels to a dataframe and pickles it.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the text data.
    labels : array-like
        The cluster labels for each data point.
    model_name : str
        The name of the model.
    p : dict
        The parameters of the model.
    feature_combination : str
        The combination of features used for clustering.
    file_path : str
        The path to the dataframe containing the text data.
    """

    if file_path == "data/processed/df_recipes_no_stopwords.pkl":
        if model_name == 'KMeans' and p['n_clusters'] == 4 and feature_combination == 'ingredients+steps+description':
            df['cluster'] = labels
            df.to_pickle(f'data/processed/cluster_labels/{model_name}.pkl')
        elif model_name == 'Agglomerative' and p['n_clusters'] == 5 and feature_combination == 'ingredients+steps+description':
            df['cluster'] = labels
            df.to_pickle(f'data/processed/cluster_labels/{model_name}.pkl')
        elif model_name == 'DBSCAN' and p['eps'] == 1 and p['min_samples'] == 10 and feature_combination == 'ingredients+steps+description':
            df['cluster'] = labels
            df.to_pickle(f'data/processed/cluster_labels/{model_name}.pkl')