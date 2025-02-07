from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import logging
import torch
import numpy as np

from models import create_model
from video_processor import VideoProcessor, VideoInfo, SceneAnalysis, AudioFeatures

@dataclass
class ClassificationResult:
    label: str
    confidence: float

@dataclass
class TaskPredictions:
    recording: List[ClassificationResult]
    content: List[ClassificationResult]
    genre: List[ClassificationResult]

@dataclass
class VideoAnalysis:
    technical: VideoInfo
    classifications: TaskPredictions
    scene_analysis: SceneAnalysis
    audio_analysis: AudioFeatures

class VideoClassificationError(Exception):
    """Custom exception for video classification errors"""
    pass

class VideoClassifier:
    """Neural network-based video classification using CLIP"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_frames: int = 32,
        top_k: int = 3
    ):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.top_k = top_k
        
        self.logger.info(f"Initializing VideoClassifier using device: {self.device}")
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        
        # Initialize model
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: Optional[str]) -> None:
        """Initialize and load the CLIP model"""
        try:
            self.model = create_model()
            
            if model_path:
                model_path = Path(model_path)
                if model_path.exists():
                    self.logger.info(f"Loading model weights from: {model_path}")
                    self.model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                else:
                    self.logger.warning(
                        f"Model path {model_path} does not exist. "
                        "Using default weights."
                    )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Cache label lists for each task using new structure
            self.recording_labels = self.model.prompts.recording
            self.content_labels = self.model.prompts.content
            self.genre_labels = self.model.prompts.genre
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise VideoClassificationError(f"Model initialization failed: {str(e)}")
    
    def _get_task_predictions(
        self,
        logits: torch.Tensor,
        labels: List[str]
    ) -> List[ClassificationResult]:
        """Convert model logits to classification results"""
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_indices = np.argsort(probs)[-self.top_k:][::-1]
        
        return [
            ClassificationResult(
                label=labels[i],
                confidence=float(probs[i])
            )
            for i in top_indices
        ]
    
    def _prepare_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Prepare frames for model input"""
        try:
            processed_frames = []
            for frame in frames:
                # Ensure frame is in RGB format
                if frame.shape[-1] != 3:
                    frame = frame[..., :3]
                processed_frames.append(frame)
            return processed_frames
            
        except Exception as e:
            self.logger.error(f"Frame preparation failed: {str(e)}")
            raise VideoClassificationError(f"Frame preparation failed: {str(e)}")
    
    def _run_model_inference(
        self,
        frames: List[np.ndarray]
    ) -> TaskPredictions:
        """Run model inference on processed frames"""
        try:
            processed_frames = self._prepare_frames(frames)
            
            with torch.no_grad():
                outputs = self.model(processed_frames)
                
                return TaskPredictions(
                    recording=self._get_task_predictions(
                        outputs.recording,  # Changed from outputs['recording']
                        self.recording_labels
                    ),
                    content=self._get_task_predictions(
                        outputs.content,    # Changed from outputs['content']
                        self.content_labels
                    ),
                    genre=self._get_task_predictions(
                        outputs.genre,      # Changed from outputs['genre']
                        self.genre_labels
                    )
                )
                
        except Exception as e:
            self.logger.error(f"Model inference failed: {str(e)}")
            raise VideoClassificationError(f"Model inference failed: {str(e)}")
    
    def classify_video(self, video_path: str) -> VideoAnalysis:
        """
        Classify a video file using CLIP-based zero-shot classification
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            VideoAnalysis containing technical info, classifications,
            scene analysis, and audio analysis
            
        Raises:
            VideoClassificationError: If classification fails
            FileNotFoundError: If video file doesn't exist
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Get video information
            video_info = self.video_processor.get_video_info(str(video_path))
            
            # Extract frames
            frames = self.video_processor.extract_frames(
                str(video_path),
                self.num_frames
            )
            
            # Run parallel analysis tasks
            with ThreadPoolExecutor() as executor:
                scene_future = executor.submit(
                    self.video_processor.analyze_scenes,
                    str(video_path)
                )
                audio_future = executor.submit(
                    self.video_processor.extract_audio_features,
                    str(video_path)
                )
                
                # Run model inference while waiting for other tasks
                classifications = self._run_model_inference(frames)
                
                # Get results from parallel tasks
                scene_analysis = scene_future.result()
                audio_analysis = audio_future.result()
            
            return VideoAnalysis(
                technical=video_info,
                classifications=classifications,
                scene_analysis=scene_analysis,
                audio_analysis=audio_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Video classification failed: {str(e)}")
            raise VideoClassificationError(f"Video classification failed: {str(e)}")

class VideoClassifierBuilder:
    """Builder class for configuring VideoClassifier instances"""
    
    def __init__(self):
        self.model_path = None
        self.num_frames = 32
        self.top_k = 3
    
    def with_model(self, model_path: str) -> 'VideoClassifierBuilder':
        self.model_path = model_path
        return self
    
    def with_frame_count(self, num_frames: int) -> 'VideoClassifierBuilder':
        self.num_frames = num_frames
        return self
    
    def with_top_k(self, top_k: int) -> 'VideoClassifierBuilder':
        self.top_k = top_k
        return self
    
    def build(self) -> VideoClassifier:
        return VideoClassifier(
            model_path=self.model_path,
            num_frames=self.num_frames,
            top_k=self.top_k
        )