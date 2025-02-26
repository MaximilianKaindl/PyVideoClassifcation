#!/usr/bin/env python3
import torch
import os
import json
import torchaudio
import torchaudio.transforms as T

def trace_clap_model(output_dir="traced_models", version='2023', use_cuda=False):
    from msclap import CLAP
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CLAP model
    model = CLAP(version=version, use_cuda=use_cuda)
    
    # Define the complete CLAP model
    class CompleteCLAP(torch.nn.Module):
        def __init__(self, clap_model):
            super().__init__()
            self.model = clap_model
            self.clap = clap_model.clap
            
        def forward(self, audio, text_tokens=None, compute_sim=True):
            # Handle different audio shapes
            if audio.dim() == 3:
                audio = audio.reshape(audio.shape[0], audio.shape[2])
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Get audio embeddings
            audio_embeddings = self.clap.audio_encoder(audio)[0]
            
            if text_tokens is None:
                return audio_embeddings
            
            # Process text tokens
            text_input = {}
            token_keys = self.model.token_keys
            for i, key in enumerate(token_keys):
                if i < len(text_tokens):
                    text_input[key] = text_tokens[i]
            
            text_embeddings = self.clap.caption_encoder(text_input)
            
            if not compute_sim:
                return audio_embeddings, text_embeddings
            
            # Compute similarity
            audio_embeddings_norm = audio_embeddings / torch.norm(audio_embeddings, dim=-1, keepdim=True)
            text_embeddings_norm = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
            
            logit_scale = self.clap.logit_scale.exp()
            similarity = logit_scale * text_embeddings_norm @ audio_embeddings_norm.T
            
            return audio_embeddings, text_embeddings, similarity.T
    
    # Create and prepare the model
    complete_model = CompleteCLAP(model)
    complete_model.eval()
    
    # Get parameters
    sample_rate = getattr(model.args, 'sampling_rate', 44100)
    duration = getattr(model.args, 'duration', 7.0)
    
    # Load real audio for tracing
    waveform, sr = torchaudio.load('resources/blues.mp3')
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    audio_sample = waveform.mean(dim=0) if waveform.size(0) > 1 else waveform.squeeze(0)
    target_length = int(sample_rate * duration)
    if audio_sample.size(0) < target_length:
        audio_sample = torch.nn.functional.pad(audio_sample, (0, target_length - audio_sample.size(0)))
    elif audio_sample.size(0) > target_length:
        audio_sample = audio_sample[:target_length]
    audio_sample = audio_sample.unsqueeze(0)
    
    # Prepare tokens
    test_labels = ["rock", "jazz", "classical", "pop", "blues"]
    processed_tokens = model.preprocess_text(test_labels)
    token_tensors = []
    for key in model.token_keys:
        if key in processed_tokens:
            token_tensors.append(processed_tokens[key])
    
    # CUDA handling
    if use_cuda and torch.cuda.is_available():
        complete_model = complete_model.cuda()
        audio_sample = audio_sample.cuda()
        token_tensors = [t.cuda() for t in token_tensors]
    
    # 1. Trace audio encoder
    class AudioEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, audio):
            return self.model(audio)
    
    audio_encoder = AudioEncoder(complete_model).eval()
    traced_audio_encoder = torch.jit.trace(audio_encoder, audio_sample)
    traced_audio_encoder.save(os.path.join(output_dir, f"traced_audio_encoder_{version}.pt"))
    
    # 2. Trace complete model
    class CompleteWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, audio, *text_tokens):
            return self.model(audio, list(text_tokens))
    
    complete_wrapper = CompleteWrapper(complete_model).eval()
    if use_cuda and torch.cuda.is_available():
        complete_wrapper = complete_wrapper.cuda()
    
    model.tokenizer.save_pretrained(output_dir)
    traced_complete_model = torch.jit.trace(complete_wrapper, (audio_sample, *token_tensors))
    traced_complete_model.save(os.path.join(output_dir, f"traced_clap_complete_{version}.pt"))

    simi = traced_complete_model.forward(audio_sample, *token_tensors)
    print(simi[2])    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trace CLAP model")
    parser.add_argument("--version", type=str, default="2023", choices=["2022", "2023", "clapcap"])
    parser.add_argument("--output_dir", type=str, default="traced_models")
    parser.add_argument("--use_cuda", action="store_true")
    
    args = parser.parse_args()
    trace_clap_model(args.output_dir, args.version, args.use_cuda)