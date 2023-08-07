from sklearn.metrics.pairwise import cosine_similarity
from semantic_clustering.utils.embeddings_utils import get_single_image_embedding, get_clip_embeddings
from PIL import Image
import matplotlib.pyplot as plt


def get_top_N_images(query, data, top_K=4, search_criterion="text"):
    """
    Get the top N most similar images based on a query.

    Args:
        query (str or array-like): Query for similarity search.
        data (pd.DataFrame): Data containing image embeddings and other relevant columns.
        top_K (int, optional): Number of top images to retrieve. Defaults to 4.
        search_criterion (str, optional): Search criterion ("text" or "image"). Defaults to "text".

    Returns:
        pd.DataFrame: Top N most similar images with relevant columns.
    """

    # Text to image Search
    if search_criterion.lower() == "text":
        query_vect = get_clip_embeddings(query)
    # Image to image Search
    else:
        query_vect = get_single_image_embedding(query)

    # Relevant columns
    relevant_cols = ["generated_text", "source_img_path", "cos_sim"]

    # Run similarity search
    data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity([query_vect], [x])[0][0])

    # Sort by cosine similarity and select top K images
    most_similar_images = data.sort_values(by='cos_sim', ascending=False)[1:top_K + 1]

    return most_similar_images[relevant_cols].reset_index()


def plot_images_by_side(top_images):
    """
    Plot images side by side with captions and similarity scores.

    Args:
        top_images (pd.DataFrame): DataFrame containing image paths, captions, and similarity scores.

    Returns:
        None
    """

    # Get the required data from the DataFrame
    index_values = list(top_images.index.values)
    list_images = [top_images.iloc[idx].source_img_path for idx in index_values]
    list_captions = [top_images.iloc[idx].generated_text for idx in index_values]
    similarity_scores = [top_images.iloc[idx].cos_sim for idx in index_values]

    # Set the number of rows and columns for the subplots
    n_row = n_col = 2

    # Create subplots with the specified number of rows and columns
    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 8))
    axs = axs.flatten()

    # Iterate over the image data and plot images with captions
    for img, ax, caption, sim_score in zip(list_images, axs, list_captions, similarity_scores):
        ax.imshow(Image.open(img).convert("RGB"))
        sim_score = 100 * float("{:.2f}".format(sim_score))
        ax.set_title(f"Caption: {caption}\nSimilarity: {sim_score}%", fontsize=8)
        ax.axis('off')

    # Adjust subplot spacing and display the plot
    plt.tight_layout()
    plt.show()
