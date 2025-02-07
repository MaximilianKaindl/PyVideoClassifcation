from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import logging
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

class TaskType(Enum):
    """Types of classification tasks"""
    RECORDING = "recording"
    CONTENT = "content"
    GENRE = "genre"

@dataclass
class TaskPrompts:
    """Prompts for each classification task"""
    recording: List[str]
    content: List[str]
    genre: List[str]

@dataclass
class ModelOutput:
    """Model output containing logits for each task"""
    recording: torch.Tensor
    content: torch.Tensor
    genre: torch.Tensor

class CLIPVideoNet(nn.Module):
    """Neural network model for multi-task video classification using CLIP"""
    
    DEFAULT_MODEL = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    
    def __init__(
        self,
        clip_model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize CLIP components
        self._initialize_clip(clip_model_name)
        
        # Define classification prompts
        self.prompts = self._initialize_prompts()
        
        # Pre-compute text features
        self.text_features = self._precompute_text_features()
        
        self.logger.info(
            f"Initialized CLIPVideoNet with model {clip_model_name} "
            f"on device {self.device}"
        )
    
    def _initialize_clip(self, model_name: str) -> None:
        """Initialize CLIP model and processor"""
        try:
            self.clip = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Freeze CLIP parameters
            for param in self.clip.parameters():
                param.requires_grad = False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize CLIP model: {str(e)}")
            raise ModelError(f"CLIP initialization failed: {str(e)}")
    
    def _initialize_prompts(self) -> TaskPrompts:
        """Initialize text prompts for each classification task"""
        return TaskPrompts(
            recording=[
                "a professional recording with high production value",
                "an amateur recording with low production value"
            ],
            content=[
                "a video showing a lecture or educational content",
                "a video showing entertainment or performance",
                "a video showing news or journalism",
                "a video showing promotional or advertising content"
            ],
            genre=[
                "a comedy video",
                "a drama video",
                "an action video",
                "a documentary video",
                "a horror video",
                "a romance video",
                "a sci-fi video",
                "a thriller video",
                "a mystery video",
                "a musical video",
                "an animation video"
            ]
        )
    
    def _compute_text_features(self, prompts: List[str]) -> torch.Tensor:
        """Compute CLIP text features for given prompts"""
        try:
            text_inputs = self.processor(
                text=prompts,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                return self.clip.get_text_features(**text_inputs)
                
        except Exception as e:
            self.logger.error(f"Failed to compute text features: {str(e)}")
            raise ModelError(f"Text feature computation failed: {str(e)}")
    
    def _precompute_text_features(self) -> Dict[TaskType, torch.Tensor]:
        """Pre-compute text features for all tasks"""
        try:
            with torch.no_grad():
                return {
                    TaskType.RECORDING: self._compute_text_features(
                        self.prompts.recording
                    ),
                    TaskType.CONTENT: self._compute_text_features(
                        self.prompts.content
                    ),
                    TaskType.GENRE: self._compute_text_features(
                        self.prompts.genre
                    )
                }
        except Exception as e:
            self.logger.error(f"Failed to precompute text features: {str(e)}")
            raise ModelError(f"Text feature precomputation failed: {str(e)}")
    
    def _process_frames(
        self,
        frames: List[np.ndarray]
    ) -> torch.Tensor:
        """Process frames through CLIP"""
        try:
            # Convert numpy arrays to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Process images through CLIP processor
            inputs = self.processor(
                images=pil_frames,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                return self.clip.get_image_features(**inputs)
                
        except Exception as e:
            self.logger.error(f"Failed to process frames: {str(e)}")
            raise ModelError(f"Frame processing failed: {str(e)}")
    
    def forward(self, frames: List[np.ndarray]) -> ModelOutput:
        """
        Forward pass for video classification
        
        Args:
            frames: List of frames as numpy arrays (RGB format)
            
        Returns:
            ModelOutput containing logits for each classification task
            
        Raises:
            ModelError: If forward pass fails
        """
        try:
            # Process frames through CLIP
            image_features = self._process_frames(frames)
            
            # Average features across frames
            averaged_features = image_features.mean(dim=0, keepdim=True)
            
            # Compute logits for each task
            with torch.no_grad():
                logit_scale = self.clip.logit_scale.exp()
                
                return ModelOutput(
                    recording=logit_scale * averaged_features @ self.text_features[
                        TaskType.RECORDING
                    ].t(),
                    content=logit_scale * averaged_features @ self.text_features[
                        TaskType.CONTENT
                    ].t(),
                    genre=logit_scale * averaged_features @ self.text_features[
                        TaskType.GENRE
                    ].t()
                )
                
        except Exception as e:
            self.logger.error(f"Forward pass failed: {str(e)}")
            raise ModelError(f"Model forward pass failed: {str(e)}")

class ModelBuilder:
    """Builder class for configuring CLIPVideoNet instances"""
    
    def __init__(self):
        self.model_name = CLIPVideoNet.DEFAULT_MODEL
        self.device = None
    
    def with_model_name(self, model_name: str) -> 'ModelBuilder':
        self.model_name = model_name
        return self
    
    def with_device(self, device: torch.device) -> 'ModelBuilder':
        self.device = device
        return self
    
    def build(self) -> nn.Module:
        return CLIPVideoNet(
            clip_model_name=self.model_name,
            device=self.device
        )

def create_model(
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Create and initialize the CLIP-based video classification model"""
    builder = ModelBuilder()
    
    if model_name:
        builder.with_model_name(model_name)
    if device:
        builder.with_device(device)
        
    return builder.build()