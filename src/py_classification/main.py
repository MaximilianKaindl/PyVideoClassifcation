from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import logging
import threading
from queue import Queue
from typing import Dict, Optional

from transcription_service import (
    TranscriptionService,
    TranscriptionResult,
    TranscriberBuilder,
    TranscriptionConfig,
    TranscriptionTask
)
from video_classifier import VideoClassifier, VideoAnalysis, VideoClassifierBuilder

class ProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

@dataclass
class ProcessingResult:
    """Container for video processing results"""
    classification: VideoAnalysis
    transcription: TranscriptionResult
    error: Optional[str] = None

class MultimediaProcessor:
    """Handles parallel processing of video files for classification and transcription"""
    
    def __init__(
        self,
        classifier: Optional[VideoClassifier] = None,
        transcriber: Optional[TranscriptionService] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with improved defaults
        self.classifier = classifier or VideoClassifierBuilder().build()
        self.transcriber = transcriber or TranscriberBuilder().\
            with_config(TranscriptionConfig(
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=True
            )).build()
        
        # Initialize thread management
        self.threads: Dict[str, threading.Thread] = {}
        self.queues: Dict[str, Queue] = {
            'transcription': Queue(),
            'classification': Queue()
        }
        
        self.logger.info("Initialized MultimediaProcessor")
    
    def _transcribe_video(self, video_path: Path) -> None:
        """Run video transcription in separate thread"""
        try:
            result = self.transcriber.transcribe(
                str(video_path),
                language="en",
                task=TranscriptionTask.TRANSCRIBE
            )
            self.queues['transcription'].put(result)
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            self.queues['transcription'].put(
                TranscriptionResult(
                    text="",
                    language="unknown",
                    error=str(e)
                )
            )
    
    def _classify_video(self, video_path: Path) -> None:
        """Run video classification in separate thread"""
        try:
            result = self.classifier.classify_video(str(video_path))
            self.queues['classification'].put(result)
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            self.queues['classification'].put({"error": str(e)})
    
    def process_video(self, video_path: str) -> ProcessingResult:
        """
        Process video for transcription and classification
        
        Args:
            video_path: Path to input video file
            
        Returns:
            ProcessingResult containing classification and transcription results
            
        Raises:
            ProcessingError: If processing fails
            FileNotFoundError: If video file doesn't exist
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Start processing threads
            self.threads['transcription'] = threading.Thread(
                target=self._transcribe_video,
                args=(video_path,)
            )
            self.threads['classification'] = threading.Thread(
                target=self._classify_video,
                args=(video_path,)
            )
            
            # Start threads
            for thread in self.threads.values():
                thread.start()
            
            # Wait for completion
            for thread in self.threads.values():
                thread.join()
            
            # Collect results
            classification_result = self.queues['classification'].get()
            transcription_result = self.queues['transcription'].get()
            
            # Check for errors
            error_messages = []
            if isinstance(classification_result, dict) and 'error' in classification_result:
                error_messages.append(f"Classification: {classification_result['error']}")
            if transcription_result.error:
                error_messages.append(f"Transcription: {transcription_result.error}")
            
            if error_messages:
                raise ProcessingError("; ".join(error_messages))
            
            return ProcessingResult(
                classification=classification_result,
                transcription=transcription_result
            )
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            return ProcessingResult(
                classification={},
                transcription=TranscriptionResult(
                    text="",
                    language="unknown"
                ),
                error=str(e)
            )

class MultimediaProcessorBuilder:
    """Builder class for configuring MultimediaProcessor instances"""
    
    def __init__(self):
        self.classifier = None
        self.transcriber = None
    
    def with_classifier(self, classifier: VideoClassifier) -> 'MultimediaProcessorBuilder':
        self.classifier = classifier
        return self
    
    def with_transcriber(self, transcriber: TranscriptionService) -> 'MultimediaProcessorBuilder':
        self.transcriber = transcriber
        return self
    
    def build(self) -> MultimediaProcessor:
        return MultimediaProcessor(
            classifier=self.classifier,
            transcriber=self.transcriber
        )

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

def main() -> None:
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create processor
        processor = MultimediaProcessorBuilder().build()
        
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
            {
                'technical': vars(results.classification.technical),
                'classifications': {
                    task: [vars(pred) for pred in preds]
                    for task, preds in vars(results.classification.classifications).items()
                },
                'scene_analysis': vars(results.classification.scene_analysis),
                'audio_analysis': vars(results.classification.audio_analysis)
            },
            indent=2
        ))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()