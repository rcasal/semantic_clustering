import argparse
import pandas as pd
from utils.utils import create_output_dirs
from utils.vision_api_utils import get_vision_api_features
import json
 

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process images and extract text')
    
    # Add arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input images directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--remove_if_exists', action='store_true', help='Remove output folder if it already exists')

    # Parse the arguments
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder
    create_output_dirs(args.output_path, args.remove_if_exists)

    # Load the JSON data from the file
    with open(f'{args.input_path}/cloud_vision_ai_assets_features.json', "r") as file:
        response_list = json.load(file)

    # Get the cloud_vision_ai_assets_features 
    df = get_vision_api_features(response_list)

    # Save the DataFrame with the added embeddings column(s)
    df.to_csv(f'{args.output_path}/ads_data.csv', index=False)


if __name__ == '__main__':
    main()

# how to call it:
# python your_script.py --input_path INPUT_PATH --output_path OUTPUT_PATH --remove_if_exists --columns column_1 column2
