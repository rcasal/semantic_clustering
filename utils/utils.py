import os
import shutil
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
import warnings
import pytesseract
from transformers import pipeline


def preprocess_text(text):
    """
    Preprocess the input text by cleaning and modifying it.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Remove special characters and replace newlines with spaces
    clean_text = text.replace('\x0c', '').replace('\n', ' ').strip()

    # Replace multiple spaces with a single space
    clean_text = ' '.join(clean_text.split())

    # Add period before capital letters after a newline
    new_text = ""
    for i in range(len(clean_text)):
        if i > 0 and clean_text[i] == ' ' and clean_text[i-1] == '\n':
            if clean_text[i+1].isupper():
                new_text += '. '
            else:
                new_text += ''
        else:
            new_text += clean_text[i]

    return new_text


def image_to_text(input_path,
                  output_path,
                  remove_if_exists=False):
    """
    Extract text from images using an image captioning model and Tesseract OCR.

    Args:
        input_path (str): The path to the input images directory.
        output_path (str): The path to the output directory for saving the results.
        remove_if_exist (bool): Whether to remove the existing output folder if it already exists.

    Raises:
        ValueError: If the output folder already exists and `remove_if_exist` is False.
    """
    warnings.filterwarnings('ignore')  # to disable warnings
    # Define input and output paths
    input_path = os.path.join(input_path)
    output_path = os.path.join(output_path)

    # Create output folder
    create_output_dirs(output_path, remove_if_exists)

    # Define the encoder model
    model_name = "Salesforce/blip-image-captioning-large"
    image_to_text_pipe = pipeline("image-to-text", model=model_name, device=0)

    # Loop through the images
    print(f'Generating text...')
    generated_texts = []
    file_names = []
    image_texts = []
    source_img_paths = []
    for filename in tqdm(os.listdir(input_path)):
        # Load the image
        if filename.endswith('.png') or filename.endswith('.jpg'):
            source_img_path = os.path.join(input_path, filename)
            img = Image.open(source_img_path)
            # Get image description
            with torch.no_grad():
                generated_text = image_to_text_pipe(img)[0]['generated_text']
            # Get image text
            image_text = preprocess_text(pytesseract.image_to_string(img))

            # Save the generated text and file name
            generated_texts.append(generated_text)
            image_texts.append(image_text)
            file_names.append(filename)
            source_img_paths.append(source_img_path)

    # Create a dataframe from the generated text and file names
    df = pd.DataFrame({'file_name': file_names, 'generated_text': generated_texts, 'ocr_text': image_texts, 'source_img_path': source_img_paths})

    # Combine the generated text and OCR text together
    df["combined"] = ("Description: " + df.generated_text.str.strip() + "; Text: " + df.ocr_text.str.strip())
   
    # Save dataframe
    print('Saving model')
    df.to_csv(os.path.join(output_path, 'ads_data.csv'), index=False)


def create_output_dirs(output_path: str, remove_if_exists: bool = False) -> None:
    """
    Create output directories and handle existing output folder.

    Args:
        output_path (str): Path to the output directory.
        remove_if_exists (bool, optional): Whether to remove the output folder if it already exists. 
                                           Defaults to False.

    Raises:
        ValueError: If the output folder already exists and `remove_if_exists` is False.
    """
    # Check if output folder already exists
    if os.path.exists(output_path):
        if remove_if_exists:
            print(f'Removing existing output folder {output_path}')
            shutil.rmtree(output_path)
        else:
            raise ValueError(f'Output folder {output_path} already exists')

    # Create output directories
    print(f"Creating directories in {output_path}")
    os.makedirs(output_path)


def fill_nan_values(df: pd.DataFrame, replace_value: str = 'Text not identified.') -> pd.DataFrame:
    """
    Replace NaN values in each column of a pandas DataFrame with a specified value and return the cleaned DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        replace_value (str, optional): Value to replace NaN values with. Defaults to 'Text not identified.'.

    Returns:
        pd.DataFrame: DataFrame with NaN values replaced.
    """
    for column in df.columns:
        # Identify NaN values in the column
        nan_indices = df[column].isna()

        # Replace NaN values with the specified value
        df.loc[nan_indices, column] = replace_value

        # Count the number of replaced values in the column
        replaced_count = nan_indices.sum()

        # Print the replaced values count for the column
        print(f"Replaced {replaced_count} NaN values in column '{column}'.")

    return df


