from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Queue
import threading
from typing import Dict, Optional
from classifier_services.transcription_service import (
    TranscriptionService,
    TranscriptionResult,
    TranscriberBuilder,
    TranscriptionConfig,
    TranscriptionTask
)
from classifier_services.clip_inference_service import VideoClassifier, VideoAnalysis, VideoClassifierBuilder

class ProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

@dataclass
class ProcessingResult:
    """Container for video processing results"""
    classification: VideoAnalysis
    transcription: TranscriptionResult
    error: Optional[str] = None

class MultimediaClassifier:
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

class MultimediaClassifierBuilder:
    """Builder class for configuring MultimediaProcessor instances"""
    
    def __init__(self):
        self.classifier = None
        self.transcriber = None
    
    def with_classifier(self, classifier: VideoClassifier) -> 'MultimediaClassifierBuilder':
        self.classifier = classifier
        return self
    
    def with_transcriber(self, transcriber: TranscriptionService) -> 'MultimediaClassifierBuilder':
        self.transcriber = transcriber
        return self
    
    def build(self) -> MultimediaClassifier:
        return MultimediaClassifier(
            classifier=self.classifier,
            transcriber=self.transcriber
        )