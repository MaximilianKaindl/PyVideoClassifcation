from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import logging
import threading
from queue import Queue
from typing import Dict, Optional

from classifier_services.transcription_service import (
    TranscriptionResult,
)
from util.multifactor_classifier import MultimediaClassifierBuilder, ProcessingResult


def setup_logging() -> None:
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process video for transcription and classification'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input video file'
    )
    return parser.parse_args()

def format_transcription_result(result: TranscriptionResult) -> Dict:
    """Format transcription result for JSON output"""
    output = {
        'text': result.text,
        'language': result.language
    }
    
    if result.chunks:
        output['chunks'] = [
            {
                'start': chunk.start,
                'end': chunk.end,
                'text': chunk.text
            }
            for chunk in result.chunks
        ]
    
    if result.error:
        output['error'] = result.error
    
    return output

def format_classification_results(results: ProcessingResult) -> Dict:
    """Format classification results for JSON output"""
    return {
        'technical': vars(results.classification.technical),
        'classifications': {
            category: [
                {
                    'label': pred.label,
                    'confidence': pred.confidence
                }
                for pred in category_preds.predictions
            ]
            for category, category_preds in results.classification.classifications.items()
        },
        'scene_analysis': vars(results.classification.scene_analysis),
        'audio_analysis': vars(results.classification.audio_analysis)
    }

def main() -> None:
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create processor
        processor = MultimediaClassifierBuilder().build()
        
        # Process video
        logger.info(f"Processing video: {args.input_path}")
        results = processor.process_video(args.input_path)
        
        if results.error:
            logger.error(f"Processing failed: {results.error}")
            return
        
        # Output results
        print("\nTranscription Results:")
        print(json.dumps(
            format_transcription_result(results.transcription),
            indent=2
        ))
        
        print("\nVideo Classification Results:")
        print(json.dumps(
            format_classification_results(results),
            indent=2
        ))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
if __name__ == "__main__":
    main()