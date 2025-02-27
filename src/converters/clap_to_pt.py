#!/usr/bin/env python3
import torch
import os
import torchaudio
import torchaudio.transforms as T
import argparse
from typing import Dict, List, Optional, Tuple, Union


class CLAPTraceWrapper(torch.nn.Module):
    """Wrapper class for CLAP model to make it traceable."""
    
    def __init__(self, clap_model, token_keys):
        super().__init__()
        self.clap = clap_model
        self.keys = token_keys
        
    def forward(self, audio, token1, token2=None):
        """Forward pass that handles the expected inputs."""
        with torch.no_grad():
            if token2 is not None:
                text_input = {self.keys[0]: token1, self.keys[1]: token2}
            else:
                text_input = {self.keys[0]: token1}
            
            return self.clap(audio, text_input)


def prepare_audio_sample(audio_path: str, sample_rate: int, duration: float) -> torch.Tensor:
    """Process an audio file into a tensor suitable for the CLAP model."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert stereo to mono if needed
    audio_sample = waveform.mean(dim=0) if waveform.size(0) > 1 else waveform.squeeze(0)
    
    # Adjust length to match target duration
    target_length = int(sample_rate * duration)
    if audio_sample.size(0) < target_length:
        audio_sample = torch.nn.functional.pad(audio_sample, (0, target_length - audio_sample.size(0)))
    elif audio_sample.size(0) > target_length:
        audio_sample = audio_sample[:target_length]
    
    # Add batch dimension
    return audio_sample.unsqueeze(0)


def prepare_text_tokens(model, test_labels: List[str], use_cuda: bool) -> Dict[str, torch.Tensor]:
    """Process text labels into token tensors for the CLAP model."""
    processed_tokens = model.preprocess_text(test_labels)
    
    # Move to CUDA if needed
    if use_cuda and torch.cuda.is_available():
        processed_tokens = {k: v.cuda() for k, v in processed_tokens.items()}
    
    return processed_tokens


def extract_token_tensors(processed_tokens: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """Extract token tensors from the processed tokens dictionary."""
    token_keys = list(processed_tokens.keys())
    input_ids = processed_tokens.get("input_ids", None)
    attention_mask = processed_tokens.get("attention_mask", None)
    
    # For models that might use different keys
    if input_ids is None and "input_ids" not in token_keys:
        print(f"Available token keys: {token_keys}")
        # Use the first available tensor
        first_key = token_keys[0]
        input_ids = processed_tokens[first_key]
        # And if there's a second one, use that too
        attention_mask = processed_tokens[token_keys[1]] if len(token_keys) > 1 else None
    
    return token_keys, input_ids.detach(), attention_mask.detach() if attention_mask is not None else None


def trace_model(wrapper: torch.nn.Module, 
                inputs: Tuple[torch.Tensor, ...], 
                output_path: str) -> torch.jit.ScriptModule:
    """Trace the model and save it to the specified path."""
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper, 
            inputs,
            check_trace=False
        )
    
    traced_model.save(output_path)
    print(f"Successfully traced and saved model to {output_path}")
    return traced_model


def test_traced_model(model_path: str, inputs: Tuple[torch.Tensor, ...], test_labels: List[str]) -> bool:
    """Test a traced model to ensure it works correctly and print similarity scores."""
    try:
        loaded_model = torch.jit.load(model_path)
        with torch.no_grad():
            outputs = loaded_model(*inputs)
        
        print("Successfully loaded and tested the traced model")
        
        # Print similarity scores
        if len(outputs) >= 3:  # Ensure we have the logit_scale and embeddings
            caption_embed, audio_embed, logit_scale = outputs
            
            # Normalize embeddings for cosine similarity
            caption_embed = caption_embed / caption_embed.norm(dim=-1, keepdim=True)
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (logit_scale * (audio_embed @ caption_embed.T)).softmax(dim=-1)
            
            print("\nSimilarity scores (higher is better):")
            for i, label in enumerate(test_labels):
                scores = similarity[0, :].cpu().numpy()  # Using [0] since we have one audio sample
                best_match_idx = scores.argmax()
                print(f"{label} (idx {i}): {scores[i]:.4f} (best match: {test_labels[best_match_idx]} with {scores[best_match_idx]:.4f})")
                
            # Print matrix form for clarity
            print("\nSimilarity matrix (audio → text):")
            similarity_np = similarity.cpu().numpy()
            for i in range(similarity_np.shape[0]):  # For each audio
                print(f"Audio sample {i}:", end=" ")
                for j in range(similarity_np.shape[1]):  # For each text
                    print(f"{test_labels[j]}: {similarity_np[i, j]:.4f}", end=", " if j < similarity_np.shape[1]-1 else "\n")
                    
        return True
    except Exception as e:
        print(f"Error testing the traced model: {e}")
        return False


def trace_clap_model(output_dir: str, version: str, use_cuda: bool, audio_path: str = 'resources/blues.mp3'):
    """Trace a CLAP model and save it to the specified directory."""
    from msclap import CLAP
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CLAP model
    model = CLAP(version=version, use_cuda=use_cuda)
    
    # Get parameters
    sample_rate = getattr(model.args, 'sampling_rate', 44100)
    duration = getattr(model.args, 'duration', 7.0)
    
    # Prepare inputs
    audio_sample = prepare_audio_sample(audio_path, sample_rate, duration)
    
    # Move to CUDA if needed
    if use_cuda and torch.cuda.is_available():
        audio_sample = audio_sample.cuda()
    
    # Prepare text tokens
    test_labels = ["rock", "jazz", "classical", "pop", "blues"]
    processed_tokens = prepare_text_tokens(model, test_labels, use_cuda)
    
    # Save tokenizer
    model.tokenizer.save_pretrained(output_dir)
    
    # Extract token tensors
    token_keys, input_ids, attention_mask = extract_token_tensors(processed_tokens)
    
    # Create the wrapper and disable gradients
    wrapper = CLAPTraceWrapper(model.clap, token_keys)
    for param in wrapper.parameters():
        param.requires_grad = False
    
    # Trace the model
    model_path = os.path.join(output_dir, f"traced_clap_{version}.pt")
    
    if attention_mask is not None:
        inputs = (audio_sample, input_ids, attention_mask)
    else:
        inputs = (audio_sample, input_ids)
    
    traced_model = trace_model(wrapper, inputs, model_path)
    
    # Test the saved model
    test_traced_model(model_path, inputs, test_labels)
    
    print("\nTesting original model for comparison:")
    with torch.no_grad():
        caption_embed, audio_embed, logit_scale = model.clap(audio_sample, processed_tokens)
        
        # Normalize embeddings for cosine similarity
        caption_embed = caption_embed / caption_embed.norm(dim=-1, keepdim=True)
        audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (logit_scale * (audio_embed @ caption_embed.T)).softmax(dim=-1)
        
        print("Original model similarity scores:")
        for i, label in enumerate(test_labels):
            scores = similarity[0, :].cpu().numpy()  # Using [0] since we have one audio sample
            best_match_idx = scores.argmax()
            print(f"{label} (idx {i}): {scores[i]:.4f} (best match: {test_labels[best_match_idx]} with {scores[best_match_idx]:.4f})")
        
        # Print matrix form
        print("\nSimilarity matrix (audio → text):")
        similarity_np = similarity.cpu().numpy()
        for i in range(similarity_np.shape[0]):  # For each audio
            print(f"Audio sample {i}:", end=" ")
            for j in range(similarity_np.shape[1]):  # For each text
                print(f"{test_labels[j]}: {similarity_np[i, j]:.4f}", end=", " if j < similarity_np.shape[1]-1 else "\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Trace CLAP model for faster inference")
    parser.add_argument("--version", type=str, default="2023", 
                        choices=["2022", "2023", "clapcap"],
                        help="CLAP model version to use")
    parser.add_argument("--output_dir", type=str, default="traced_models",
                        help="Directory to save traced models")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA for tracing if available")
    parser.add_argument("--audio_path", type=str, default="resources/blues.mp3",
                        help="Path to audio file for tracing")
    
    args = parser.parse_args()
    trace_clap_model(args.output_dir, args.version, args.use_cuda, args.audio_path)


if __name__ == "__main__":
    main()