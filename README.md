# VoiceGeneration

Voice cloning and generation toolkit using Coqui TTS.

## Prerequisites

1. Python 3.8 or newer
2. FFmpeg installed and on PATH
   ```powershell
   # Install with Chocolatey (recommended)
   choco install ffmpeg
   
   # Or download from https://ffmpeg.org/download.html
   # Add the bin/ folder to your PATH
   ```
3. Optional but recommended: NVIDIA GPU with CUDA installed

## Quick Setup

1. Clone this repository and navigate to it
   ```powershell
   cd "c:\Users\Dell\OneDrive\VoiceGeneration\repo"
   ```

2. Create and activate a Python virtual environment
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```

   If you have an NVIDIA GPU, also install CUDA support:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Quick Demo - Voice Cloning

The quickest way to try voice cloning is using our demo script:

```powershell
# Activate the environment if not already active
.\.venv\Scripts\Activate.ps1

# Run the demo (replace with your video/audio file path)
python scripts/demo_voice_clone.py --input "path/to/your/video.mp4" --text "Hello, this is a test."
```

This will:
1. Extract audio if you provided a video file
2. Process the audio (trim silence, normalize)
3. Download a pre-trained TTS model
4. Generate speech in the target voice

Outputs will be saved in the `outputs/` directory.

### Audio Processing Utilities

The `src/utils/audio.py` module provides utilities for:
- Extracting audio from video
- Processing audio files (resampling, normalization)
- Loading TTS models
- Generating speech

Example usage in Python:

```python
from src.utils.audio import extract_audio, prepare_audio, load_voice_model

# Extract audio from video
audio_path = extract_audio("input.mp4")

# Process for TTS
processed_path = prepare_audio(audio_path)

# Load model and generate
synthesizer = load_voice_model()
wav = synthesizer.tts(
    text="Hello world",
    speaker_wav=str(processed_path)
)
```

## Project Structure

- `src/` - Source code
  - `utils/` - Utility functions
    - `audio.py` - Audio processing utilities
- `scripts/` - Command line tools
  - `demo_voice_clone.py` - Voice cloning demo
- `tests/` - Unit tests
- `requirements.txt` - Python dependencies

## Common Issues

1. FFmpeg not found
   - Make sure FFmpeg is installed and in your PATH
   - Try running `ffmpeg -version` in terminal

2. CUDA/GPU errors
   - If you don't have a GPU, the code will run on CPU (slower)
   - With GPU, make sure CUDA toolkit is installed
   - Install PyTorch with CUDA support (see setup steps)

3. Audio processing errors
   - Check input file exists and is valid audio/video
   - Try processing shorter clips first
   - Ensure enough disk space for extracted audio

## License

MIT License - See LICENSE file
