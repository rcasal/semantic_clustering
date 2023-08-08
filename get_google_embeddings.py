import argparse
from utils.utils import create_output_dirs
from utils.google_embeddings_utils import getImageEmbeddingFromGcsObject, EmbeddingPredictionClient
from google.cloud import storage
import csv
import os


def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process images and extract text')
    
    # Add arguments
    parser.add_argument('--input_bucket', type=str, required=True, help='Path to the input images bucket')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--remove_if_exists', action='store_true', help='Remove output folder if it already exists')
    parser.add_argument('--project_id', type=str, required=True, help='GCP Project ID')

    # Parse the arguments
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder
    create_output_dirs(args.output_path, args.remove_if_exists)

    # This is the GCS bucket that holds the images that you want to analyze and
    # index. You will need the bucket list and object reading permission to proceed.
    # The default bucket provided here contains 61 images contributed by the
    # engineer team. If you want to try your own image set, feel free to point this
    # to another GCS bucket that holds your images. Please make sure all files in
    # the GCS bucket are images (e.g. JPG, PNG). Non image files would cause
    # inference exception down below.
    IMAGE_SET_BUCKET_NAME = args.input_bucket # @param {type: "string"}
    PROJECT_ID = args.project_id

    client = EmbeddingPredictionClient(project=PROJECT_ID)
    gcsBucket = storage.Client().get_bucket(IMAGE_SET_BUCKET_NAME)

    output_file_path = os.path.join(args.output_path, 'ads_data.csv')
    with open(output_file_path, 'w') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(['gcsUri', 'embedding'])
        for blob in gcsBucket.list_blobs():
            gcsUri = "gs://" + IMAGE_SET_BUCKET_NAME + "/" + blob.name
            print("Processing {}".format(gcsUri))
            embedding = getImageEmbeddingFromGcsObject(IMAGE_SET_BUCKET_NAME, blob.name, client)
            csvWriter.writerow([gcsUri, str(embedding)])


if __name__ == '__main__':
    main()

# how to call it:
# python your_script.py --input_path INPUT_PATH --output_path OUTPUT_PATH --remove_if_exists --columns column_1 column2
