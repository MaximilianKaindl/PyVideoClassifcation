from msclap import CLAP
import torch
import json

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap_model = CLAP(version = '2023', use_cuda=True)

torch.jit.save(clap_model.clap, "clap_scripted.pt")
# Save tokenizer as JSON
tokenizer_json = clap_model.tokenizer.to_json()
with open("tokenizer.json", "w") as json_file:
    json.dump(tokenizer_json, json_file)

# Sample class labels
class_labels = ["rock", "jazz", "classical", "pop", "blues"]

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(class_labels)

file_paths = ['resources/blues.mp3']
# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(file_paths)

# Compute similarity between audio and text embeddings 
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

# Apply softmax to similarities
softmax_similarities = torch.nn.functional.softmax(torch.tensor(similarities), dim=1).cpu().numpy()

print(softmax_similarities)