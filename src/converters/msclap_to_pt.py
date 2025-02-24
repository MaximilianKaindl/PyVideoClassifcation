from msclap import CLAP
import torch
import json
import os

def save_clap_model():
    print("Loading CLAP model...")
    clap_model = CLAP(version='2023', use_cuda=True)
    
    # Put model in eval mode
    clap_model.clap.eval()
    
    try:
        # Save the entire state dict
        state_dict = clap_model.clap.state_dict()
        torch.save({
            'state_dict': state_dict,
            'model_config': {
                'audioenc_name': clap_model.args.audioenc_name,
                'sample_rate': clap_model.args.sampling_rate,
                'window_size': clap_model.args.window_size,
                'hop_size': clap_model.args.hop_size,
                'mel_bins': clap_model.args.mel_bins,
                'fmin': clap_model.args.fmin,
                'fmax': clap_model.args.fmax,
                'classes_num': clap_model.args.num_classes,
                'out_emb': clap_model.args.out_emb,
                'text_model': clap_model.args.text_model,
                'transformer_embed_dim': clap_model.args.transformer_embed_dim,
                'd_proj': clap_model.args.d_proj
            }
        }, 'clap_model_full.pt')
        
        # Create output directory for tokenizer
        os.makedirs('tokenizer', exist_ok=True)
        
        # Save tokenizer files
        clap_model.tokenizer.save_pretrained('tokenizer')
        print("Successfully saved CLAP model components!")
        
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    save_clap_model()