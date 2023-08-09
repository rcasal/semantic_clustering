import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib


import hdbscan
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import os


class HDBSCANClustering:
    def clusterize(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Perform HDBSCAN clustering on the specified columns of the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names to perform clustering on.

        Returns:
            pd.DataFrame: DataFrame with cluster labels.
        """
        # Convert string representations of arrays to actual arrays
        for column in columns:
            df[column] = df[column].apply(lambda x: np.array(eval(x)))

        # Concatenate the embedding columns into a new column
        df['embedding_concatenated'] = df[columns].apply(lambda x: np.hstack(x.values.tolist()), axis=1)

        # Select the concatenated embedding arrays
        self.data = np.vstack(df['embedding_concatenated'].values)

        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        df['cluster_label'] = clusterer.fit_predict(self.data)

        return df

    def visualize_clusters(self, df: pd.DataFrame):
        """
        Visualize the clusters using t-SNE.

        Args:
            df (pd.DataFrame): DataFrame with cluster labels.
        """
        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(self.data)

        unique_labels = np.unique(df['cluster_label'])

        colors = plt.cm.get_cmap('tab20', len(unique_labels))

        color_indices = df['cluster_label'].values

        plt.scatter(vis_dims[:, 0], vis_dims[:, 1], c=color_indices, cmap=colors, alpha=0.3)
        plt.title("Clusters Visualized using t-SNE")
        plt.show()

    def visualize_images(self, df: pd.DataFrame, input_path: str):
        """
        Visualize the first 10 images from each cluster.

        Args:
            df (pd.DataFrame): DataFrame with cluster labels.
            input_path (str): Path to the folder containing the images.
        """
        clusters = df.groupby('cluster_label')

        for label, cluster in clusters:
            image_names = cluster.head(10)['file_name'].tolist()

            fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

            for row in range(2):
                for col in range(5):
                    if not image_names:
                        axs[row, col].axis('off')
                        break

                    image_name = image_names.pop(0)
                    image_path = os.path.join(input_path, image_name)
                    img = Image.open(image_path)
                    axs[row, col].imshow(img)

            plt.suptitle(f'Cluster {label}', fontsize=20)
            plt.show()


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
        self.data = np.vstack(df['embedding_concatenated'].values)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        df['cluster_label'] = kmeans.fit_predict(self.data)

        return df

    def visualize_clusters(df: pd.DataFrame, n_components: int = 2):
        """
        Visualize the clusters using t-SNE.

        Args:
            df (pd.DataFrame): DataFrame with cluster labels.
            n_components (int): Number of components for t-SNE (2 or 3).
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components should be either 2 or 3")

        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=n_components, perplexity=15, random_state=42, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(df.data)

        colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]

        x = [x for x, y, _ in vis_dims] if n_components == 3 else [x for x, _ in vis_dims]
        y = [y for x, y, _ in vis_dims] if n_components == 3 else [y for _, y in vis_dims]

        color_indices = df.cluster_label.values - 1

        if n_components == 2:
            colormap = plt.cm.get_cmap('viridis', len(colors))
            plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
            plt.title("Campaigns visualized in language using t-SNE (2D)")
            plt.colorbar()
            plt.show()
        else:  # n_components == 3
            colormap = plt.cm.get_cmap('viridis', len(colors))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, vis_dims[:, 2], c=color_indices, cmap=colormap, alpha=0.3)
            ax.set_title("Campaigns visualized in language using t-SNE (3D)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.colorbar()
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
