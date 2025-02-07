from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional
import torch
import librosa
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Pipeline
)

class TranscriptionError(Exception):
    """Custom exception for transcription-related errors"""
    pass

class TranscriptionTask(Enum):
    """Types of transcription tasks"""
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"

@dataclass
class TranscriptionChunk:
    """Represents a chunk of transcribed text with timestamps"""
    start: float
    end: float
    text: str

@dataclass
class TranscriptionResult:
    """Container for transcription results"""
    text: str
    language: str
    chunks: Optional[List[TranscriptionChunk]] = None
    error: Optional[str] = None

@dataclass
class TranscriptionConfig:
    """Configuration for transcription processing"""
    chunk_length_s: int = 30
    batch_size: int = 8
    return_timestamps: bool = True
    sample_rate: int = 16000

class TranscriptionService:
    """Service for handling audio transcription using HuggingFace's Whisper models"""
    
    DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        config: Optional[TranscriptionConfig] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Set device and model configurations
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.config = config or TranscriptionConfig()
        
        # Initialize model components
        self.processor = None
        self.model = None
        self.pipe: Optional[Pipeline] = None
        
        self.logger.info(
            f"Initialized TranscriptionService using device: {self.device}"
        )
    
    def _load_models(self) -> None:
        """Load Whisper models and processor"""
        if self.pipe is not None:
            return
            
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Set model configuration
            model_kwargs = {"attn_implementation": "sdpa"}
            torch_dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Initialize pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch_dtype,
                model_kwargs=model_kwargs
            )
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
                device_map="auto"
            )
            
            self.logger.info("Whisper models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise TranscriptionError(f"Model loading failed: {str(e)}")
    
    def _extract_audio(self, video_path: Path, output_path: Path) -> None:
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path: Path to input video
            output_path: Path to output audio file
            
        Raises:
            TranscriptionError: If audio extraction fails
        """
        try:
            cmd = [
                "./ffmpeg_tools/ffmpeg",
                '-i', str(video_path),
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',
                '-ar', str(self.config.sample_rate),
                '-ac', '1',  # Mono audio
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"FFmpeg audio extraction failed: {str(e)}")
            raise TranscriptionError(f"Audio extraction failed: {str(e)}")
    
    def transcribe(
        self,
        video_path: str,
        language: Optional[str] = None,
        task: TranscriptionTask = TranscriptionTask.TRANSCRIBE
    ) -> TranscriptionResult:
        """
        Transcribe audio from a video file
        
        Args:
            video_path: Path to the video file
            language: Language code (e.g., 'en', 'fr'). None for auto-detection
            task: TranscriptionTask.TRANSCRIBE or TRANSLATE
            
        Returns:
            TranscriptionResult containing transcription data
            
        Raises:
            TranscriptionError: If transcription fails
            FileNotFoundError: If video file doesn't exist
        """
        # Ensure models are loaded
        self._load_models()
        
        # Validate video path
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_path = Path(temp_audio.name)
            
            try:
                # Extract audio to temporary file
                self._extract_audio(video_path, temp_path)
                
                # Load audio file
                self.logger.info("Loading audio file")
                speech_array, _ = librosa.load(
                    temp_path,
                    sr=self.config.sample_rate
                )
                
                # Prepare generation kwargs
                generate_kwargs = {"task": task.value}
                if language:
                    generate_kwargs["language"] = language
                
                # Run transcription
                self.logger.info(
                    f"Transcribing audio in {self.config.chunk_length_s}s chunks"
                )
                result = self.pipe(
                    speech_array,
                    chunk_length_s=self.config.chunk_length_s,
                    batch_size=self.config.batch_size,
                    return_timestamps=self.config.return_timestamps,
                    generate_kwargs=generate_kwargs
                )
                
                # Process chunks if available
                chunks = None
                if self.config.return_timestamps and "chunks" in result:
                    chunks = [
                        TranscriptionChunk(
                            start=chunk['timestamp'][0],
                            end=chunk['timestamp'][1],
                            text=chunk['text'].strip()
                        )
                        for chunk in result["chunks"]
                    ]
                
                # Create result object
                return TranscriptionResult(
                    text=result['text'].strip(),
                    language=result.get('language', language or 'unknown'),
                    chunks=chunks
                )
                
            except Exception as e:
                self.logger.error(f"Transcription failed: {str(e)}")
                return TranscriptionResult(
                    text="",
                    language="unknown",
                    error=str(e)
                )
                
            finally:
                try:
                    temp_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete temp file: {str(e)}")

class TranscriberBuilder:
    """Builder class for configuring TranscriptionService instances"""
    
    def __init__(self):
        self.model_name = TranscriptionService.DEFAULT_MODEL
        self.device = None
        self.config = TranscriptionConfig()
    
    def with_model(self, model_name: str) -> 'TranscriberBuilder':
        self.model_name = model_name
        return self
    
    def with_device(self, device: str) -> 'TranscriberBuilder':
        self.device = device
        return self
    
    def with_config(self, config: TranscriptionConfig) -> 'TranscriberBuilder':
        self.config = config
        return self
    
    def build(self) -> TranscriptionService:
        return TranscriptionService(
            model_name=self.model_name,
            device=self.device,
            config=self.config
        )