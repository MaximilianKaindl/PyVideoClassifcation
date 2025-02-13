import torch
from safetensors.torch import load_file
import argparse
from huggingface_hub import hf_hub_download
from transformers import CLIPModel
import os
from collections import OrderedDict
import open_clip
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def export_clip_model(model_name, dataset_name, output_path):
    """
    Export a CLIP model to TorchScript format with encode_image and encode_text methods
    """
    try:
        print(f"Loading CLIP model {model_name}...")

        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=dataset_name
        )
        model.eval()
        scripted_model = torch.jit.script(model)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        scripted_model.save(output_path)
        
        loaded_model = torch.jit.load(output_path)
        tokenizer = open_clip.get_tokenizer(model_name)

        return loaded_model, preprocess, tokenizer
        
    except Exception as e:
        print(f"Error during export: {str(e)}")
        raise

def classify_image_text(model, preprocess, tokenizer, image_path, candidate_texts):
    """
    Classify an image against a list of candidate text descriptions using CLIP.
    
    Args:
        model: The loaded CLIP model
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
        image_path: Path to the input image
        candidate_texts: List of text descriptions to compare against
    
    Returns:
        Dictionary of text descriptions and their similarity scores
    """
    # Load and preprocess image
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0)
    
    # Encode image
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Encode text descriptions
    text_tokens = tokenizer(candidate_texts)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Create results dictionary
    results = {}
    for text, score in zip(candidate_texts, similarity[0].tolist()):
        results[text] = score
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export CLIP model and classify images')
    parser.add_argument('model_name', type=str, help='Name of the CLIP model')
    parser.add_argument('dataset_name', type=str, help='Name of the training Dataset Name')
    parser.add_argument('output_path', type=str, help='Path for output TorchScript model')
    parser.add_argument('image_path', type=str, help='Path to the image for classification')
    
    args = parser.parse_args()
    
    # Export and load the model
    model, preprocess, tokenizer = export_clip_model(args.model_name, args.dataset_name, args.output_path)
    
    # Example candidate text descriptions
    candidate_texts = [
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a car",
        "a photo of a house",
        "a photo of a person"
    ]
    
    # Perform classification
    print("\nClassifying image...")
    results = classify_image_text(model, preprocess, tokenizer, args.image_path, candidate_texts)
    
    # Print results sorted by confidence
    print("\nClassification Results:")
    for text, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{text}: {score:.2%} confidence")