import argparse
from utils.utils import image_to_text

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process images and extract text')
    
    # Add arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input images directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--remove_if_exists', action='store_true', help='Remove output folder if it already exists')
    parser.add_argument('--model_name', default='CLIP', help='Remove output folder if it already exists')

    # Parse the arguments
    return parser.parse_args()

def main():
    
    args = parse_args()

    # Call the image_to_text function with the provided arguments
    image_to_text(
        input_path=args.input_path, 
        output_path=args.output_path, 
        remove_if_exists=args.remove_if_exists,
        model_name=args.model_name
    )


if __name__ == '__main__':
    main()
