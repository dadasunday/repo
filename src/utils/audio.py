"""Audio processing utilities for voice generation/cloning."""

import os
from pathlib import Path
from typing import Optional, Union

import ffmpeg
import librosa
import soundfile as sf
import torch
from TTS.utils.synthesizer import Synthesizer


def extract_audio(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 22050,
    mono: bool = True,
) -> Path:
    """Extract audio from video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save WAV file (defaults to same name as video)
        sample_rate: Target sample rate (22050 Hz is common for TTS)
        mono: Convert to mono if True
    
    Returns:
        Path to extracted WAV file
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stream = ffmpeg.input(str(video_path))
    stream = ffmpeg.output(
        stream,
        str(output_path),
        acodec="pcm_s16le",
        ac=1 if mono else 2,
        ar=str(sample_rate),
    )
    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    
    return output_path


def load_voice_model(model_path: Optional[str] = None) -> Synthesizer:
    """Load a Coqui TTS model for inference.
    
    If no model_path provided, downloads and uses the default model.
    For list of available models see: https://github.com/coqui-ai/TTS#models
    
    Args:
        model_path: Path to local model or name of pretrained model
        
    Returns:
        Loaded synthesizer ready for inference
    """
    # If no model specified, use a decent default
    if model_path is None:
        model_name = "tts_models/multilingual/multi-dataset/your_tts"
        return Synthesizer(
            model_name=model_name,
        )
    
    # Load local model
    return Synthesizer(
        model_path=model_path,
    )


def prepare_audio(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_sr: int = 22050,
    trim_silence: bool = True,
    normalize: bool = True,
) -> Path:
    """Prepare audio file for TTS training/inference.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save processed audio (defaults to input_processed.wav)
        target_sr: Target sample rate
        trim_silence: Remove silence from start/end
        normalize: Normalize audio volume
        
    Returns:
        Path to processed audio file
    """
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
    else:
        output_path = Path(output_path)
        
    # Load audio
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Process
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
    if normalize:
        y = librosa.util.normalize(y)
        
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y, target_sr)
    
    return output_path