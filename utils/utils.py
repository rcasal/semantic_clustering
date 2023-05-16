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


def image_to_text(input_path='campaign_assets_rc/CampaignAssets/MicrosoftSurfacePro9',
                  output_path='/content/text',
                  remove_if_exist=True):
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

    # Define the encoder model
    model_name = "Salesforce/blip-image-captioning-large"
    image_to_text_pipe = pipeline("image-to-text", model=model_name, device=0)

    # Check if output folder already exists
    if os.path.exists(output_path):
        if remove_if_exist:
            print(f'Removing existing output folder {output_path}')
            shutil.rmtree(output_path)
        else:
            raise ValueError(f'Output folder {output_path} already exists')

    # Create output directories
    print(f"Creating directories in {output_path}")
    os.makedirs(output_path)

    # Loop through the images
    print(f'Generating text...')
    generated_texts = []
    file_names = []
    image_texts = []
    for filename in tqdm(os.listdir(input_path)):
        # Load the image
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = Image.open(os.path.join(input_path, filename))
            # Get image description
            with torch.no_grad():
                generated_text = image_to_text_pipe(img)[0]['generated_text']
            # Get image text
            image_text = preprocess_text(pytesseract.image_to_string(img))

            # Save the generated text and file name
            generated_texts.append(generated_text)
            image_texts.append(image_text)
            file_names.append(filename)

    # Create a dataframe from the generated text and file names
    df = pd.DataFrame({'file_name': file_names, 'generated_text': generated_texts, 'image_text': image_texts})

    # Save dataframe
    print('Saving model')
    df.to_csv('text/text_dataframe.csv', index=False)
