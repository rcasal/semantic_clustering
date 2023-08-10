import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import webcolors
from typing import Tuple

    


def get_vision_api_features(response_list) -> pd.DataFrame:
# Preprocess the data and create a list of dictionaries
    processed_data = []
    for creative_data in response_list:
        creative_id = creative_data["creative_id"]
        creative_uri = creative_data["creative_uri"]
        
        # Text annotations
        text_annotation, text_annotation_centroid = get_text_annotations(creative_data)
        
        # Object annotations
        object_annotations_name = [d['name'] for d in creative_data["localized_object_annotations"] if d["score"] > 0.65]
        
        # Logo annotations
        logo_annotation, logo_annotation_centroid = get_logo_annotations(creative_data)
        
        # Face annotations
        face_annotations = get_face_annotations(creative_data)
        faces_amount = face_annotations['faces_amount']
        joy_likelihood = face_annotations['joy_likelihood']
        sorrow_likelihood = face_annotations['sorrow_likelihood']
        anger_likelihood = face_annotations['anger_likelihood']
        surprise_likelihood = face_annotations['surprise_likelihood']
        under_exposed_likelihood = face_annotations['under_exposed_likelihood']
        blurred_likelihood = face_annotations['blurred_likelihood']
        headwear_likelihood = face_annotations['headwear_likelihood']

        # Perform preprocessing on creative_data search_safe_annotations
        adult = creative_data["search_safe_annotations"][0]["adult"]
        spoof =  creative_data["search_safe_annotations"][0]["spoof"]
        medical =  creative_data["search_safe_annotations"][0]["medical"]
        violence =  creative_data["search_safe_annotations"][0]["violence"]
        racy =  creative_data["search_safe_annotations"][0]["racy"]
        
        # Perform preprocessing on creative_data color
        rgb_list = [[d['red'], d['green'], d['blue']] for d in creative_data["dominant_color_annotations"] if d['score'] > 0.1] # pixel fraction or score?
        color_names = [closest_colour(rgb) for rgb in rgb_list]

        # Appending data
        processed_data.append({
            "creative_id": creative_id,
            "creative_uri": creative_uri,
            "text_annotation": text_annotation,
            "text_annotation_centroid": text_annotation_centroid,
            "object_annotations_name": object_annotations_name,
            "logo_annotation": logo_annotation,
            "logo_annotation_centroid": logo_annotation_centroid,
            "faces_amount": faces_amount,
            "joy_likelihood": joy_likelihood,
            "sorrow_likelihood": sorrow_likelihood,
            "anger_likelihood": anger_likelihood,
            "surprise_likelihood": surprise_likelihood,
            "under_exposed_likelihood": under_exposed_likelihood,
            "blurred_likelihood": blurred_likelihood,
            "headwear_likelihood": headwear_likelihood,
            "adult": adult,
            "spoof": spoof,
            "medical": medical,
            "violence": violence,
            "racy": racy,
            "color_names": color_names
        })

    # Create a DataFrame from the processed data
    return pd.DataFrame(processed_data)


def centroid(v0: float, v1: float, v2: float, v3: float) -> float:
    """
    Calculates the centroid of four values.
    
    Args:
        v0 (float): First vertice value.
        v1 (float): Second vertice value.
        v2 (float): Third vertice value.
        v3 (float): Fourth vertice value.
    
    Returns:
        float: Centroid value.
    """
    return (v0 + v1 + v2 + v3) / 4


def closest_colour(requested_colour: Tuple[int, int, int]) -> str:
    """
    Finds the closest color name to the requested RGB color.
    
    Args:
        requested_colour (Tuple[int, int, int]): RGB values of the requested color.
    
    Returns:
        str: Closest color name.
    """
    min_colours = {}
    
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    
    return min_colours[min(min_colours.keys())]


def get_text_annotations(creative_data):
    # Perform preprocessing on creative_data text annotations
    aux_ta = creative_data["text_annotations"][0]
    text_annotation = aux_ta["description"]
    text_annotation_centroid = [
        centroid(aux_ta["x0"],aux_ta["x1"],aux_ta["x2"],aux_ta["x3"]),
        centroid(aux_ta["y0"],aux_ta["y1"],aux_ta["y2"],aux_ta["y3"]),
    ]
    """
    Comento esto, porque el text annotation viene con todo el texto en 
    el primer elemento, y después tiene desglosado por bloques.
    """
    #descriptions = [d['description'] for d in aux[1:]] 
    #concatenated_description = ' '.join(descriptions)

    return text_annotation, text_annotation_centroid


def get_logo_annotations(creative_data):
    # Perform preprocessing on creative_data logo annotation
    aux_la = creative_data["logo_annotations"][0]
    logo_annotation = aux_la["description"]
    logo_annotation_centroid = [
        centroid(aux_la["x0"],aux_la["x1"],aux_la["x2"],aux_la["x3"]),
        centroid(aux_la["y0"],aux_la["y1"],aux_la["y2"],aux_la["y3"]),
    ]

    return logo_annotation, logo_annotation_centroid


def get_face_annotations(creative_data):   
    faces_amount = sum(1 for d in creative_data["face_annotations"] if d['detection_confidence'] > 0.65)

    # Calculate the sums for each sentiment likelihood over elements with detection confidence > 0.7
    joy_likelihood = 0.0
    sorrow_likelihood = 0.0
    anger_likelihood = 0.0
    surprise_likelihood = 0.0
    under_exposed_likelihood = 0.0
    blurred_likelihood = 0.0
    headwear_likelihood = 0.0   
    if faces_amount > 0:
        for d in  creative_data["face_annotations"]:
            if d['detection_confidence'] > 0.65:
                joy_likelihood += d['joy_likelihood']
                sorrow_likelihood += d['sorrow_likelihood']
                anger_likelihood += d['anger_likelihood']
                surprise_likelihood += d['surprise_likelihood']
                under_exposed_likelihood += d['under_exposed_likelihood']
                blurred_likelihood += d['blurred_likelihood']
                headwear_likelihood += d['headwear_likelihood']

        # Calculate the averages for each sentiment likelihood
        joy_likelihood = joy_likelihood / faces_amount
        sorrow_likelihood = sorrow_likelihood / faces_amount
        anger_likelihood = anger_likelihood / faces_amount
        surprise_likelihood = surprise_likelihood / faces_amount
        under_exposed_likelihood = under_exposed_likelihood / faces_amount
        blurred_likelihood = blurred_likelihood / faces_amount
        headwear_likelihood = headwear_likelihood / faces_amount
    
    return {
        'faces_amount': faces_amount,
        'joy_likelihood': joy_likelihood,
        'sorrow_likelihood': sorrow_likelihood,
        'anger_likelihood': anger_likelihood,
        'surprise_likelihood': surprise_likelihood,
        'under_exposed_likelihood': under_exposed_likelihood,
        'blurred_likelihood': blurred_likelihood,
        'headwear_likelihood': headwear_likelihood
    }