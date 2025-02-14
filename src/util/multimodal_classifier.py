from dataclasses import dataclass
from pathlib import Path
import json
import logging
import tempfile
import threading
from queue import Queue
from typing import Dict, List, Optional

from util.ffmpeg_video_processor import (
    FFMPEGVideoProcessor,
    VideoInfo,
    SceneAnalysis,
    AudioFeatures
)
from util.labels import Labels, get_default_labels, parse_labels_file

@dataclass
class ClassificationResult:
    """Result of classification for a single input against category labels"""
    category: str
    label: str
    probability: float

@dataclass
class VideoAnalysis:
    """Container for video analysis results"""
    technical: VideoInfo
    classifications: List[ClassificationResult]
    scene_analysis: SceneAnalysis
    audio_analysis: AudioFeatures

@dataclass
class ProcessingResult:
    """Container for video processing results"""
    classification: VideoAnalysis
    error: Optional[str] = None

class ProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

class MultimodalClassifier:
    """Handles parallel processing of video files for classification"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        labels_file: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels_file = labels_file
        
        # Initialize components
        self.video_processor = FFMPEGVideoProcessor()
        self.labels = parse_labels_file(labels_file) if labels_file else get_default_labels()
        
        # Initialize thread management
        self.threads: Dict[str, threading.Thread] = {}
        self.queues: Dict[str, Queue] = {
            'classification': Queue()
        }
        
        self.logger.info("Initialized MultimediaClassifier")
    
    def _classify_video(self, video_path: Path) -> None:
        """Run video classification in separate thread"""
        try:
            results: List[ClassificationResult] = []
            
            # Get video technical info
            technical_info = self.video_processor.get_video_info(str(video_path))
            
            # Run scene analysis
            scene_analysis = self.video_processor.analyze_scenes(str(video_path))
            
            # Run audio analysis
            audio_analysis = self.video_processor.extract_audio_features(str(video_path))
            
            # Run classification for each category
            for category, category_labels in self.labels.categories.items():
                self.logger.info(f"Processing category: {category}")
                
                with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                    temp_output = f.name
                
                try:
                    # Run CLIP classification
                    self.video_processor.run_clip_classification(
                        str(video_path),
                        self.model_path,
                        self.tokenizer_path,
                        category_labels,
                        temp_output
                    )
                    
                    # Parse results
                    best_label, probability = self.video_processor.parse_clip_output(temp_output)
                    if best_label:
                        results.append(ClassificationResult(
                            category=category,
                            label=best_label,
                            probability=probability
                        ))
                        
                finally:
                    if Path(temp_output).exists():
                        Path(temp_output).unlink()
            
            # Create video analysis result
            analysis = VideoAnalysis(
                technical=technical_info,
                classifications=results,
                scene_analysis=scene_analysis,
                audio_analysis=audio_analysis
            )
            
            self.queues['classification'].put(analysis)
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            self.queues['classification'].put({"error": str(e)})
    
    def process_video(self, video_path: str) -> ProcessingResult:
        """
        Process video for classification
        
        Args:
            video_path: Path to input video file
            
        Returns:
            ProcessingResult containing classification results
            
        Raises:
            ProcessingError: If processing fails
            FileNotFoundError: If video file doesn't exist
        """
        if not self.model_path or not self.tokenizer_path:
            raise ValueError("Model path and tokenizer path must be provided")
            
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
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
            
            # Check for errors
            error_messages = []
            if isinstance(classification_result, dict) and 'error' in classification_result:
                error_messages.append(f"Classification: {classification_result['error']}")
            
            if error_messages:
                raise ProcessingError("; ".join(error_messages))
            
            return ProcessingResult(
                classification=classification_result,
            )
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            return ProcessingResult(
                classification=VideoAnalysis(
                    technical=VideoInfo(
                        width=0, height=0, duration=0.0,
                        bitrate=0, fps=0.0, codec="",
                        has_audio=False, audio_codec=None, size=0
                    ),
                    classifications=[],
                    scene_analysis=SceneAnalysis(
                        scene_changes=0,
                        scene_timestamps=[],
                        status="error"
                    ),
                    audio_analysis=AudioFeatures(
                        sample_rate=0,
                        channels=0,
                        duration=0.0,
                        bit_depth=0,
                        error="Processing failed"
                    )
                ),
                error=str(e)
            )

class MultimodalClassifierBuilder:
    """Builder class for configuring MultimediaProcessor instances"""
    
    def __init__(self):
        self.model_path = None
        self.tokenizer_path = None
        self.labels_file = None
    
    def with_model(self, model_path: str) -> 'MultimodalClassifierBuilder':
        self.model_path = model_path
        return self
    
    def with_tokenizer(self, tokenizer_path: str) -> 'MultimodalClassifierBuilder':
        self.tokenizer_path = tokenizer_path
        return self
    
    def with_labels_file(self, labels_file: str) -> 'MultimodalClassifierBuilder':
        self.labels_file = labels_file
        return self
    
    def build(self) -> MultimodalClassifier:
        return MultimodalClassifier(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            labels_file=self.labels_file,
        )