import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embeddings(text):
        # Tokenize input text and convert to PyTorch tensors
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

        # Obtain embeddings
        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_states = outputs.last_hidden_state

        # Get the embedding of the first token
        embedding = last_hidden_states[0][0]

        return embedding.tolist()

    for column in columns:
        df[f'embedding_{column}'] = df[column].apply(get_bert_embeddings)

    return df

