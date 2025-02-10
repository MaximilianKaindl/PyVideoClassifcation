import tempfile
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import logging

from util.labels import Labels, parse_labels_file
from util.video_processor import FFMPEGVideoProcessor

@dataclass
class ClassificationResult:
    """Result of classification for a single input against category labels"""
    category: str
    label: str
    probability: float

def setup_logging() -> None:
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main(
    ffmpeg_bin: str,
    input_file: str,
    model_file: str,
    tokenizer_file: str,
    labels_file: str
) -> None:
    """Main function to run video classification"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting classification process for: {input_file}")
    logger.info(f"Using FFmpeg binary: {ffmpeg_bin}")
    logger.info(f"Model file: {model_file}")
    logger.info(f"Tokenizer file: {tokenizer_file}")
    logger.info(f"Labels file: {labels_file}\n")
    
    # Parse labels
    labels = parse_labels_file(labels_file)
    results: List[ClassificationResult] = []
    
    # Initialize video processor
    processor = FFMPEGVideoProcessor(ffmpeg_path=ffmpeg_bin)
    
    # Run classification for each category
    for category, category_labels in labels.categories.items():
        logger.info(f"Processing category: {category}")
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_output = f.name
        
        try:
            # Run CLIP classification
            processor.run_clip_classification(
                input_file,
                model_file,
                tokenizer_file,
                category_labels,
                temp_output
            )
            
            # Parse results
            best_label, probability = processor.parse_clip_output(temp_output)
            if best_label:
                results.append(ClassificationResult(
                    category=category,
                    label=best_label,
                    probability=probability
                ))
                logger.info(
                    f"Best match for {category}: {best_label} "
                    f"(probability: {probability:.2f})"
                )
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    # Create output filename
    input_path = Path(input_file)
    output_file = input_path.with_stem(f"{input_path.stem}_tagged")
    
    # Format metadata
    metadata = {
        result.category: {
            'label': result.label,
            'probability': result.probability
        }
        for result in results
    }
    
    # Write metadata
    logger.info("\nWriting final output with metadata...")
    processor.write_clip_metadata(input_file, str(output_file), metadata)
    logger.info(f"Success! Classifications written to: {output_file}")
    logger.info(f"Final classifications: {metadata}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg', required=True, help='Path to FFmpeg binary')
    parser.add_argument('--input', required=True, help='Input video/image file')
    parser.add_argument('--model', required=True, help='CLIP model file (.pt)')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer file')
    parser.add_argument('--labels', required=True, help='Labels file')
    
    setup_logging()
    args = parser.parse_args()
    main(args.ffmpeg, args.input, args.model, args.tokenizer, args.labels)