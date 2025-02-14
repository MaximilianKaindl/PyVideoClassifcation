from pathlib import Path
import argparse
import json
import logging
from typing import Dict, Any

from util.multimodal_classifier import MultimodalClassifierBuilder
from util.ffmpeg_video_processor import VideoInfo, SceneAnalysis, AudioFeatures

def setup_logging() -> None:
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process video for classification'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to CLIP model file (.pt)',
        default="resources/models/openclip-vit-l-14.pt"
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Path to tokenizer file',
        default="resources/tokenizer.json"
    )
    parser.add_argument(
        '--labels',
        type=str,
        help='Path to labels file',
        default="resources/labels.txt"
    )
    return parser.parse_args()

def format_video_info(info: VideoInfo) -> Dict[str, Any]:
    """Format video technical info for JSON output"""
    return {
        'width': info.width,
        'height': info.height,
        'duration': info.duration,
        'bitrate': info.bitrate,
        'fps': info.fps,
        'codec': info.codec,
        'has_audio': info.has_audio,
        'audio_codec': info.audio_codec,
        'size': info.size
    }

def format_scene_analysis(analysis: SceneAnalysis) -> Dict[str, Any]:
    """Format scene analysis for JSON output"""
    return {
        'scene_changes': analysis.scene_changes,
        'scene_timestamps': analysis.scene_timestamps,
        'status': analysis.status,
        'error': analysis.error
    }

def format_audio_analysis(analysis: AudioFeatures) -> Dict[str, Any]:
    """Format audio analysis for JSON output"""
    return {
        'sample_rate': analysis.sample_rate,
        'channels': analysis.channels,
        'duration': analysis.duration,
        'bit_depth': analysis.bit_depth,
        'error': analysis.error
    }

def format_classification_results(results: Any) -> Dict[str, Any]:
    """Format classification results for JSON output"""
    return {
        'technical': format_video_info(results.classification.technical),
        'classifications': [
            {
                'category': result.category,
                'label': result.label,
                'probability': result.probability
            }
            for result in results.classification.classifications
        ],
        'scene_analysis': format_scene_analysis(results.classification.scene_analysis),
        'audio_analysis': format_audio_analysis(results.classification.audio_analysis)
    }

def main() -> None:
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate input paths
        for path_arg in [args.input_path, args.model, args.tokenizer]:
            if not Path(path_arg).exists():
                raise FileNotFoundError(f"File not found: {path_arg}")
        
        if args.labels and not Path(args.labels).exists():
            raise FileNotFoundError(f"Labels file not found: {args.labels}")
        
        # Create processor
        processor = MultimodalClassifierBuilder()\
            .with_model(args.model)\
            .with_tokenizer(args.tokenizer)\
            .with_labels_file(args.labels if args.labels else None)\
            .build()
        
        # Process video
        logger.info(f"Processing video: {args.input_path}")
        results = processor.process_video(args.input_path)
        
        if results.error:
            logger.error(f"Processing failed: {results.error}")
            return
        
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