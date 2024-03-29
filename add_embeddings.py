import argparse
import pandas as pd
from utils.utils import create_output_dirs, fill_nan_values
from utils.embeddings_utils import add_bert_embeddings, add_clip_embeddings, add_image_embeddings

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process images and extract text')
    
    # Add arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input images directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--remove_if_exists', action='store_true', help='Remove output folder if it already exists')
    parser.add_argument('--columns', nargs='+', required=True, help='List of column names for BERT embeddings')

    # Parse the arguments
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder
    create_output_dirs(args.output_path, args.remove_if_exists)

    # Read the CSV file with generated text and image text
    df = pd.read_csv(f'{args.input_path}/ads_data.csv')
    
    # Drop NaN
    df = fill_nan_values(df)

    # Get the columns for BERT embeddings
    columns_bert = args.columns
    
    # Call the add_bert_embeddings function
    df_with_embeddings = add_bert_embeddings(df, columns_bert)

    # Call the add_clip_embeddings function
    df_with_embeddings = add_clip_embeddings(df_with_embeddings, 'generated_text') # there's an error with the lenght
    
    # Call the add_image_embeddings function
    df_with_embeddings = add_image_embeddings(df_with_embeddings)
    
    # Save the DataFrame with the added embeddings column(s)
    df_with_embeddings.to_csv(f'{args.output_path}/ads_data.csv', index=False)


if __name__ == '__main__':
    main()

# how to call it:
# python your_script.py --input_path INPUT_PATH --output_path OUTPUT_PATH --remove_if_exists --columns column_1 column2
