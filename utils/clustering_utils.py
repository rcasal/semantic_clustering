import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

class KMeansClustering:
    def clusterize(self, df: pd.DataFrame, columns: list, num_clusters: int) -> pd.DataFrame:
        """
        Perform K-means clustering on the specified columns of the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names to perform clustering on.
            num_clusters (int): Number of clusters to create.

        Returns:
            pd.DataFrame: DataFrame with cluster labels.
        """
        # Convert string representations of arrays to actual arrays
        for column in columns:
            df[column] = df[column].apply(lambda x: np.array(eval(x)))

        # Concatenate the embedding columns into a new column
        df['embedding_concatenated'] = df[columns].apply(lambda x: np.hstack(x.values.tolist()), axis=1)

        # Select the concatenated embedding arrays
        data = np.vstack(df['embedding_concatenated'].values)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        df['cluster_label'] = kmeans.fit_predict(data)

        return df

    def visualize_clusters(self, df: pd.DataFrame):
        """
        Visualize the clusters using t-SNE.

        Args:
            df (pd.DataFrame): DataFrame with cluster labels.
        """
        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(data)

        colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]

        x = [x for x, y in vis_dims]
        y = [y for x, y in vis_dims]
        color_indices = df.cluster_label.values - 1

        colormap = matplotlib.colors.ListedColormap(colors)
        plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
        plt.title("Campaigns visualized in language using t-SNE")
        plt.show()

    def visualize_images(self, df: pd.DataFrame, input_path: str):
        """
        Visualize the first 10 images from each cluster.

        Args:
            df (pd.DataFrame): DataFrame with cluster labels.
            input_path (str): Path to the folder containing the images.
        """
        grupos = df.groupby('cluster_label')

        for i, grupo in grupos:
            imagenes = grupo.head(10)['file_name'].tolist()

            fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

            for r in range(2):
                for c in range(5):
                    imagen = imagenes.pop(0)

                    img = Image.open(os.path.join(input_path, imagen))
                    axs[r, c].imshow(img)

                    if not imagenes:
                        axs[r, c].axis('off')
                        break

            plt.suptitle(f'Cluster {i}', fontsize=20)
            plt.show()
