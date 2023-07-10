import torch
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel, CLIPTokenizer
import pandas as pd
from PIL import Image
import numpy as np


def add_bert_embeddings(df: pd.DataFrame, columns: str or list) -> pd.DataFrame:
    """
    Add BERT embeddings to the specified column(s) of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or list): Name of the column(s) to add embeddings to.

    Returns:
        pd.DataFrame: DataFrame with the added embeddings column(s).
    """       
    # Data type sanity check
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, list):
        columns = columns
    else:
        raise ValueError("Invalid column parameter. Expected str or list.")

    # Load pre-trained model and tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embeddings(text):
        # Tokenize input text and convert to PyTorch tensors
        input_ids = torch.tensor([bert_tokenizer.encode(text, add_special_tokens=True)])

        # Obtain embeddings
        with torch.no_grad():
            outputs = bert_model(input_ids)
            last_hidden_states = outputs.last_hidden_state

        # Get the embedding of the first token
        embedding = last_hidden_states[0][0]

        return embedding.tolist()

    for column in columns:
        print(f'Processing {column}...')
        df[f'bert_embedding_{column}'] = df[column].apply(get_bert_embeddings)

    return df


def get_clip_embeddings(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define the model ID
    model_ID = "openai/clip-vit-base-patch32"
    # Load pre-trained model and tokenizer
    clip_model = CLIPModel.from_pretrained(model_ID)#.to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    # Tokenize input text and convert to PyTorch tensors
    inputs = clip_tokenizer(text, return_tensors="pt")
    text_embeddings = clip_model.get_text_features(**inputs)
    # convert the embeddings to a numpy array
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np


def add_clip_embeddings(df: pd.DataFrame, columns: str or list) -> pd.DataFrame:
    """
    Add CLIP embeddings to the specified column(s) of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or list): Name of the column(s) to add embeddings to.

    Returns:
        pd.DataFrame: DataFrame with the added embeddings column(s).
    """       
    # Data type sanity check
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, list):
        columns = columns
    else:
        raise ValueError("Invalid column parameter. Expected str or list.")
    
    for column in columns:
        print(f'Processing {column}...')
        if column != 'combined':
            df[f'clip_embedding_{column}'] = df[column].apply(get_clip_embeddings)

    return df

def get_single_image_embedding(source_img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ID = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_ID)#.to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_ID)
    image = clip_processor(
        text = None,
        images = Image.open(source_img_path).convert("RGB"),
        return_tensors="pt"
        )["pixel_values"]#.to(device)
    embedding = clip_model.get_image_features(image)

    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np

def add_image_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df["img_embeddings"] = df['souce_img_path'].apply(get_single_image_embedding)
    df['img_embeddings'] = df['img_embeddings'].apply(lambda x: np.array(x.strip('[]').split(), dtype=np.float64))
    df['img_embeddings'] = df['img_embeddings'].apply(lambda x: x.reshape(1, -1))
    
    return df
