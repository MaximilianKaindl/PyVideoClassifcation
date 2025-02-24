from msclap import CLAP
from msclap.models.clap import CLAP as CLAPModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import torchaudio
import torchaudio.transforms as T
import json
import random

def load_and_test_clap():
    print("Loading saved model and tokenizer...")
    
    # Load the saved model state and config
    saved_data = torch.load('clap_model_full.pt')
    model_config = saved_data['model_config']
    
    # Create a new CLAP model instance with the saved config
    model = CLAPModel(
        audioenc_name=model_config['audioenc_name'],
        sample_rate=model_config['sample_rate'],
        window_size=model_config['window_size'],
        hop_size=model_config['hop_size'],
        mel_bins=model_config['mel_bins'],
        fmin=model_config['fmin'],
        fmax=model_config['fmax'],
        classes_num=model_config['classes_num'],
        out_emb=model_config['out_emb'],
        text_model=model_config['text_model'],
        transformer_embed_dim=model_config['transformer_embed_dim'],
        d_proj=model_config['d_proj']
    )
    
    # Load the state dict
    model.load_state_dict(saved_data['state_dict'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('tokenizer')
    
    print("Testing model with sample inputs...")
    
    # Sample class labels from your original script
    class_labels = ["rock", "jazz", "classical", "pop", "blues"]
    
    # Create a wrapper class to match your original usage
    class ModelWrapper:
        def __init__(self, model, tokenizer, config):
            self.clap = model
            self.tokenizer = tokenizer
            self.config = config
            self.use_cuda = torch.cuda.is_available()
            self.sample_rate = config['sample_rate']
            self.duration = 10  # Default duration in seconds
        
        def preprocess_text(self, text_queries):
            if 'gpt' in self.config['text_model'].lower():
                text_queries = [text + ' <|endoftext|>' for text in text_queries]
                
            encoded = self.tokenizer(
                text_queries,
                padding='max_length',
                max_length=77,  # Default CLAP text length
                truncation=True,
                return_tensors="pt"
            )
            
            if self.use_cuda:
                encoded = {k: v.cuda() for k, v in encoded.items()}
                
            return encoded
        
        def load_audio_into_tensor(self, audio_path):
            # Load audio file
            audio_time_series, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                audio_time_series = resampler(audio_time_series)
            
            # Convert to mono if necessary
            if audio_time_series.shape[0] > 1:
                audio_time_series = torch.mean(audio_time_series, dim=0, keepdim=True)
            
            # Reshape to match expected input
            audio_time_series = audio_time_series.reshape(-1)
            
            # Handle duration
            target_length = int(self.sample_rate * self.duration)
            
            if len(audio_time_series) > target_length:
                # If longer, take a random segment
                start = random.randint(0, len(audio_time_series) - target_length)
                audio_time_series = audio_time_series[start:start + target_length]
            else:
                # If shorter, pad with zeros
                padding_length = target_length - len(audio_time_series)
                audio_time_series = F.pad(audio_time_series, (0, padding_length))
            
            if self.use_cuda:
                audio_time_series = audio_time_series.cuda()
            
            return audio_time_series.unsqueeze(0)  # Add batch dimension
        
        def get_text_embeddings(self, text_queries):
            preprocessed_text = self.preprocess_text(text_queries)
            with torch.no_grad():
                return self.clap.caption_encoder(preprocessed_text)
        
        def get_audio_embeddings(self, audio_path):
            audio_tensor = self.load_audio_into_tensor(audio_path)
            with torch.no_grad():
                return self.clap.audio_encoder(audio_tensor)[0]
        
        def compute_similarity(self, audio_embeddings, text_embeddings):
            audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=-1, keepdim=True)
            text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
            
            logit_scale = self.clap.logit_scale.exp()
            similarity = logit_scale * text_embeddings @ audio_embeddings.T
            return similarity.T
    
    wrapper = ModelWrapper(model, tokenizer, model_config)
    
    # Process text embeddings
    print("Computing text embeddings...")
    text_embeddings = wrapper.get_text_embeddings(class_labels)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Process audio embeddings
    print("\nComputing audio embeddings...")
    audio_embeddings = wrapper.get_audio_embeddings('resources/blues.mp3')
    print(f"Audio embeddings shape: {audio_embeddings.shape}")
    
    # Compute similarities
    print("\nComputing similarities...")
    similarities = wrapper.compute_similarity(audio_embeddings, text_embeddings)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(similarities, dim=1)
    
    # Print results
    print("\nResults:")
    for i, (label, prob) in enumerate(zip(class_labels, probabilities[0].cpu().detach().numpy())):
        print(f"{label}: {prob*100:.2f}%")

if __name__ == "__main__":
    load_and_test_clap()