from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import logging
import re
import subprocess
import threading
from typing import Dict, List, Optional
import numpy as np

class HWAccelType(Enum):
    NVIDIA = "nvidia"
    CPU = "cpu"

@dataclass
class HWAccelConfig:
    decode: str
    encode: str
    scale: str
    
    @classmethod
    def from_type(cls, accel_type: HWAccelType) -> 'HWAccelConfig':
        if accel_type == HWAccelType.NVIDIA:
            return cls(decode='', encode='', scale='scale')
        return cls(decode='', encode='', scale='scale')

@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    bitrate: int
    fps: float
    codec: str
    has_audio: bool
    audio_codec: Optional[str]
    size: int

@dataclass
class SceneAnalysis:
    scene_changes: int
    scene_timestamps: List[float]
    status: str
    error: Optional[str] = None

@dataclass
class AudioFeatures:
    sample_rate: int
    channels: int
    duration: float
    bit_depth: int
    error: Optional[str] = None

class FFmpegError(Exception):
    """Custom exception for FFmpeg-related errors"""
    pass

class VideoProcessor:
    """Handles video processing operations using FFmpeg"""
    
    FRAME_READ_TIMEOUT = 10  # seconds
    DEFAULT_FRAME_SIZE = (224, 224)
    BUFFER_SIZE = 10 * 1024 * 1024  # 10MB
    SCENE_CHANGE_THRESHOLD = 0.3
    
    def __init__(
        self,
        ffmpeg_path: str = './ffmpeg_tools/ffmpeg',
        ffprobe_path: str = './ffmpeg_tools/ffprobe'
    ):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.logger = logging.getLogger(__name__)
        self._validate_paths()
        self.hw_config = self._detect_hw_accel()
    
    def _validate_paths(self) -> None:
        """Validate FFmpeg tool paths exist"""
        if not self.ffmpeg_path:
            raise FileNotFoundError(f"FFmpeg not found at: {self.ffmpeg_path}")
        if not self.ffprobe_path:
            raise FileNotFoundError(f"FFprobe not found at: {self.ffprobe_path}")
    
    def _detect_hw_accel(self) -> HWAccelConfig:
        """Detect available hardware acceleration"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return HWAccelConfig.from_type(HWAccelType.NVIDIA)
        except (subprocess.SubprocessError, FileNotFoundError):
            return HWAccelConfig.from_type(HWAccelType.CPU)
    
    def _run_ffprobe(self, input_path: str, args: List[str]) -> Dict:
        """Run ffprobe with given arguments and return JSON output"""
        cmd = [
            str(self.ffprobe_path),
            '-v', 'quiet',
            '-print_format', 'json',
            *args,
            str(input_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"FFprobe failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise FFmpegError(f"Failed to parse FFprobe output: {e}")
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract video metadata using FFprobe"""
        try:
            info = self._run_ffprobe(
                video_path,
                ['-show_format', '-show_streams']
            )
            
            video_stream = next(
                (s for s in info['streams'] if s['codec_type'] == 'video'),
                None
            )
            if not video_stream:
                raise FFmpegError("No video stream found")
            
            audio_stream = next(
                (s for s in info['streams'] if s['codec_type'] == 'audio'),
                None
            )
            
            return VideoInfo(
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                duration=float(info['format'].get('duration', 0)),
                bitrate=int(info['format'].get('bit_rate', 0)),
                fps=eval(video_stream.get('r_frame_rate', '0/1')),
                codec=video_stream.get('codec_name', ''),
                has_audio=audio_stream is not None,
                audio_codec=audio_stream.get('codec_name') if audio_stream else None,
                size=int(info['format'].get('size', 0))
            )
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def analyze_scenes(self, video_path: str) -> SceneAnalysis:
        """Detect scene changes using FFmpeg"""
        cmd = [
            str(self.ffmpeg_path),
            '-i', str(video_path),
            '-vf', f"select='gt(scene,{self.SCENE_CHANGE_THRESHOLD})',showinfo",
            '-f', 'null',
            '-'
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=self.BUFFER_SIZE
            )
            
            _, stderr = process.communicate()
            
            scene_times = []
            for line in stderr.split('\n'):
                if 'pts_time' in line and 'scene_score' in line:
                    match = re.search(
                        r'pts_time:([\d.]+).*scene_score:([\d.]+)',
                        line
                    )
                    if match:
                        time, score = map(float, match.groups())
                        if score > self.SCENE_CHANGE_THRESHOLD:
                            scene_times.append(time)
            
            return SceneAnalysis(
                scene_changes=len(scene_times),
                scene_timestamps=scene_times,
                status='success'
            )
            
        except Exception as e:
            self.logger.error(f"Scene analysis failed: {str(e)}")
            return SceneAnalysis(
                scene_changes=0,
                scene_timestamps=[],
                status='error',
                error=str(e)
            )
    
    def extract_audio_features(self, video_path: str) -> AudioFeatures:
        """Extract audio features using FFmpeg"""
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Use a fixed path in our temp directory
        temp_wav = temp_dir / "temp_audio.wav"
        
        try:
            # Extract audio to WAV
            cmd = [
                str(self.ffmpeg_path),
                *self.hw_config.decode.split(),
                '-i', str(Path(video_path).resolve()),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-y',
                str(temp_wav.resolve())
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Analyze audio properties
            audio_info = self._run_ffprobe(
                str(temp_wav),
                ['-show_format', '-show_streams', '-select_streams', 'a:0']
            )
            
            stream = audio_info['streams'][0]
            return AudioFeatures(
                sample_rate=int(stream.get('sample_rate', 0)),
                channels=int(stream.get('channels', 0)),
                duration=float(audio_info['format'].get('duration', 0)),
                bit_depth=int(stream.get('bits_per_sample', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Audio feature extraction failed: {str(e)}")
            return AudioFeatures(
                sample_rate=0,
                channels=0,
                duration=0.0,
                bit_depth=0,
                error=str(e)
            )
            
        finally:
            # Clean up temp file
            try:
                if temp_wav.exists():
                    temp_wav.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete temp file: {str(e)}")
    
    @contextmanager
    def _timeout_context(self, seconds: int, error_message: str = "Operation timed out"):
        """Context manager for timing out operations"""
        timeout_event = threading.Event()
        timer = threading.Timer(seconds, timeout_event.set)
        timer.daemon = True
        
        try:
            timer.start()
            yield timeout_event
        finally:
            timer.cancel()
            timer.join()
    
    def _read_frame_data(
        self,
        process: subprocess.Popen,
        frame_size: int
    ) -> Optional[bytes]:
        """Read frame data from process with timeout protection"""
        result = {"data": None, "error": None}
        
        def read_target():
            try:
                result["data"] = process.stdout.read(frame_size)
            except Exception as e:
                result["error"] = e
        
        read_thread = threading.Thread(target=read_target)
        read_thread.daemon = True
        read_thread.start()
        
        with self._timeout_context(self.FRAME_READ_TIMEOUT) as timeout_event:
            read_thread.join(self.FRAME_READ_TIMEOUT)
            
            if timeout_event.is_set() or read_thread.is_alive():
                self.logger.warning("Frame read timeout occurred")
                return None
            
            if result["error"]:
                raise result["error"]
            
            return result["data"]
    
    def extract_frames(self, video_path: str, num_frames: int) -> List[np.ndarray]:
        """Extract evenly spaced frames from video"""
        frames = []
        process = None
        
        try:
            # Get video info for frame interval calculation
            video_info = self.get_video_info(video_path)
            frame_interval = max(
                1,
                int(video_info.duration * video_info.fps / num_frames)
            )
            
            # Build FFmpeg command
            cmd = [
                str(self.ffmpeg_path),
                '-loglevel', 'quiet',
                *self.hw_config.decode.split(),
                '-i', str(video_path),
                '-vf', (
                    f"{self.hw_config.scale}="
                    f"{self.DEFAULT_FRAME_SIZE[0]}:{self.DEFAULT_FRAME_SIZE[1]},"
                    f"fps={frame_interval}/1"
                ),
                '-vsync', '0',
                '-frame_pts', '1',
                *self.hw_config.encode.split(),
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                '-'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.BUFFER_SIZE
            )
            
            frame_size = (
                self.DEFAULT_FRAME_SIZE[0] *
                self.DEFAULT_FRAME_SIZE[1] *
                3  # RGB channels
            )
            
            while len(frames) < num_frames:
                raw_frame = self._read_frame_data(process, frame_size)
                
                if raw_frame is None or not raw_frame:  # Timeout or EOF
                    break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((*self.DEFAULT_FRAME_SIZE, 3))
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}")
            raise
            
        finally:
            if process:
                try:
                    process.stdout.close()
                    process.stderr.close()
                    process.terminate()
                except Exception as e:
                    self.logger.error(f"Process cleanup failed: {str(e)}")