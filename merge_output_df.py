import argparse
import pandas as pd
from utils.utils import create_output_dirs
import os

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process images and extract text')
    
    # Add arguments
    parser.add_argument('--google_embedding_path', type=str, required=True, help='Path to the Google embeddings csv')
    parser.add_argument('--vision_api_path', type=str, required=True, help='Path to the Google embeddings csv')
    parser.add_argument('--embedding_text_path', type=str, required=True, help='Path to the embeddings csv')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--remove_if_exists', action='store_true', help='Remove output folder if it already exists')

    # Parse the arguments
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder
    create_output_dirs(args.output_path, args.remove_if_exists)

    # Read the dataframes
    df1 = pd.read_csv(os.path.join(args.embedding_text_path, 'ads_data.csv')) 
    df2 = pd.read_csv(os.path.join(args.google_embedding_path, 'ads_data.csv')) 
    df3 = pd.read_csv(os.path.join(args.vision_api_path, 'ads_data.csv')) 

    # Merge df2 and df3 using 'gcsUri' and 'creative_id'
    df2_and_df3 = pd.merge(df2, df3, left_on='gcsUri', right_on='creative_id', how='inner')

    # Drop the 'gcsUri' column from df2_and_df3
    df2_and_df3.drop(columns=['gcsUri'], inplace=True)
    print(f"{len(df2_and_df3)}/{len(df1)} rows were joined based on 'gcsUri' and 'creative_id'.")

    # Merge df1 with df2_and_df3 using a partial match on 'file_name' and 'creative_uri'
    df = pd.merge(df1, df2_and_df3, left_on=df1['file_name'], right_on=df2_and_df3['creative_uri'].str.split('/').str[-1], how='inner').drop(columns=['key_0'])

    # Print the number of rows in each join
    print(f"{len(df)}/{len(df1)} rows were joined based on 'file_name'.")

    # Save the DataFrame with the added embeddings column(s)
    df.to_csv(f'{args.output_path}/ads_data.csv', index=False)


if __name__ == '__main__':
    main()

# how to call it:
# python your_script.py --input_path INPUT_PATH --output_path OUTPUT_PATH --remove_if_exists --columns column_1 column2
