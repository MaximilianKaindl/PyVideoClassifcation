import subprocess
import tempfile
import os
import re

def log_subprocess_output(cmd, stdout, stderr):
    """Log subprocess command and its output."""
    print("\n" + "="*80)
    print("Executing command:", " ".join(cmd))
    if stdout:
        print("\nStandard output:")
        print(stdout.decode('utf-8', errors='replace'))
    if stderr:
        print("\nStandard error:")
        print(stderr.decode('utf-8', errors='replace'))
    print("="*80 + "\n")

def parse_labels_file(labels_file):
    categories = {}
    current_category = None
    
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_category = line[1:-1]
                categories[current_category] = []
            elif line and current_category:
                categories[current_category].append(line)
    
    return categories

def parse_classification_output(output_file):
    max_prob = 0
    best_label = None
    
    with open(output_file, 'r') as f:
        for line in f:
            label, prob = line.strip().split(': ')
            prob = float(prob)
            if prob > max_prob:
                max_prob = prob
                best_label = label
    
    # Remove prefix like "a video recorded with" or "a video of"
    if best_label:
        best_label = re.sub(r'^a (?:video|photo) (?:recorded with|of) ', '', best_label)
        best_label = re.sub(r'^an? ', '', best_label)
    
    return best_label, max_prob

def run_clip_classification(ffmpeg_bin, input_file, model_file, tokenizer_file, labels, temp_output):
    labels_str = '\n'.join(labels)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(labels_str)
        temp_labels = f.name

    cmd = [
        ffmpeg_bin,
        '-i', input_file,
        '-vf', f'dnn_clip=dnn_backend=torch:model={model_file}:device=cuda:labels={temp_labels}:tokenizer={tokenizer_file},avgclass=output_file={temp_output}',
        '-f', 'null', '-'
    ]
    
    try:
        print(f"\nRunning CLIP classification for input: {input_file}")
        result = subprocess.run(cmd, capture_output=True)
        log_subprocess_output(cmd, result.stdout, result.stderr)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
            
    finally:
        os.unlink(temp_labels)

def main(ffmpeg_bin, input_file, model_file, tokenizer_file, labels_file):
    print(f"Starting classification process for: {input_file}")
    print(f"Using FFmpeg binary: {ffmpeg_bin}")
    print(f"Model file: {model_file}")
    print(f"Tokenizer file: {tokenizer_file}")
    print(f"Labels file: {labels_file}\n")
    
    categories = parse_labels_file(labels_file)
    results = {}
    
    # Run classification for each category
    for category, labels in categories.items():
        print(f"\nProcessing category: {category}")
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            run_clip_classification(ffmpeg_bin, input_file, model_file, tokenizer_file, labels, temp_output)
            best_label, probability = parse_classification_output(temp_output)
            if best_label:
                results[category] = best_label
                print(f"Best match for {category}: {best_label} (probability: {probability:.2f})")
        finally:
            os.unlink(temp_output)
    
    # Create output filename
    output_file = os.path.splitext(input_file)[0] + '_tagged' + os.path.splitext(input_file)[1]
    
    # Format metadata
    tags = '; '.join([f'{cat}: {label}' for cat, label in results.items()])
    
    # Write metadata using FFmpeg
    cmd = [
        ffmpeg_bin,
        '-i', input_file,
        '-metadata', f'comment={tags}',
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-y', output_file
    ]
    
    print("\nWriting final output with metadata...")
    result = subprocess.run(cmd, capture_output=True)
    log_subprocess_output(cmd, result.stdout, result.stderr)
    
    if result.returncode == 0:
        print(f"\nSuccess! Classifications written to: {output_file}")
        print("Final results:", results)
    else:
        print("\nError writing output file!")
        raise subprocess.CalledProcessError(result.returncode, cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg', required=True, help='Path to FFmpeg binary')
    parser.add_argument('--input', required=True, help='Input video/image file')
    parser.add_argument('--model', required=True, help='CLIP model file (.pt)')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer file')
    parser.add_argument('--labels', required=True, help='Labels file')
    
    args = parser.parse_args()
    main(args.ffmpeg, args.input, args.model, args.tokenizer, args.labels)