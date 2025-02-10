# PyVideoClassifcation

PyVideoClassifcation is a Python project for classifying videos using different (CLIP) models. This project includes tools to classify videos using FFMPEG and Python scripts, as well as converting models to `.pt` format.

## Features

- **FFMPEG Classify Video**: Classify videos using FFMPEG CLIP Filter.
- **Python Classify Video**: Classify videos with Python (CLIP and Whisper Transcription).
- **Model Conversion**: Convert models to `.pt` format for usage with ffmpeg.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/MaximilianKaindl/PyVideoClassifcation.git
    cd PyVideoClassifcation
    ```

2. Install the required Python packages with the conda env file:

## Usage

### FFMPEG Classify Video

```sh
python src/ffmpeg_classification/classify_movie.py --ffmpeg ./ffmpeg_tools/ffmpeg --input resources/cartoon.mp4 --model resources/models/openclip-vit-l-14.pt --tokenizer resources/tokenizer.json --labels resources/labels.txt
```

### Python Classify Video

This will automatically download the default models laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K and openai/whisper-large-v3-turbo from Huggingface

```sh
python src/py_classification/main.py resources/cartoon.mp4
```

### Sample Model Conversion to .pt

#### Convert ViT-B-32

```sh
python src/converters/clip_to_pt.py ViT-B-32 datacomp_xl_s13b_b90k out/openclip-vit-b-32.pt resources/input.png
```

#### Convert ViT-L-14

```sh
python src/converters/clip_to_pt.py ViT-L-14 datacomp_xl_s13b_b90k out/openclip-vit-l-14.pt resources/input.png
```
