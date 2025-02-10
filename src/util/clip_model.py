from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import logging
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

from util.labels import Labels, get_default_labels, parse_labels_file

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

@dataclass
class ModelOutput:
    """Model output containing logits for each category"""
    logits: Dict[str, torch.Tensor]

class CLIPVideoNet(nn.Module):
    """Neural network model for multi-task video classification using CLIP"""
    
    DEFAULT_MODEL = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    
    def __init__(
        self,
        clip_model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
        labels_file: Optional[str] = None
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize CLIP components
        self._initialize_clip(clip_model_name)
        
        # Initialize labels
        self.labels = (
            parse_labels_file(labels_file) if labels_file
            else get_default_labels()
        )
        
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
    
    def _precompute_text_features(self) -> Dict[str, torch.Tensor]:
        """Pre-compute text features for all categories"""
        try:
            with torch.no_grad():
                return {
                    category: self._compute_text_features(labels)
                    for category, labels in self.labels.categories.items()
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
            ModelOutput containing logits for each classification category
            
        Raises:
            ModelError: If forward pass fails
        """
        try:
            # Process frames through CLIP
            image_features = self._process_frames(frames)
            
            # Average features across frames
            averaged_features = image_features.mean(dim=0, keepdim=True)
            
            # Compute logits for each category
            with torch.no_grad():
                logit_scale = self.clip.logit_scale.exp()
                category_logits = {}
                
                for category, text_features in self.text_features.items():
                    category_logits[category] = logit_scale * averaged_features @ text_features.t()
                
                return ModelOutput(logits=category_logits)
                
        except Exception as e:
            self.logger.error(f"Forward pass failed: {str(e)}")
            raise ModelError(f"Model forward pass failed: {str(e)}")

class ModelBuilder:
    """Builder class for configuring CLIPVideoNet instances"""
    
    def __init__(self):
        self.model_name = CLIPVideoNet.DEFAULT_MODEL
        self.device = None
        self.labels_file = None
    
    def with_model_name(self, model_name: str) -> 'ModelBuilder':
        self.model_name = model_name
        return self
    
    def with_device(self, device: torch.device) -> 'ModelBuilder':
        self.device = device
        return self
    
    def with_labels_file(self, labels_file: str) -> 'ModelBuilder':
        self.labels_file = labels_file
        return self
    
    def build(self) -> nn.Module:
        return CLIPVideoNet(
            clip_model_name=self.model_name,
            device=self.device,
            labels_file=self.labels_file
        )

def create_model(
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    labels_file: Optional[str] = None
) -> nn.Module:
    """Create and initialize the CLIP-based video classification model"""
    builder = ModelBuilder()
    
    if model_name:
        builder.with_model_name(model_name)
    if device:
        builder.with_device(device)
    if labels_file:
        builder.with_labels_file(labels_file)
        
    return builder.build()