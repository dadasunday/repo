import os
from pathlib import Path
import av
import librosa
import soundfile as sf
import numpy as np

def extract_audio_from_video(video_path, output_dir):
    """Extract audio from video and save as wav file."""
    print(f"Extracting audio from {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path for the WAV file
    output_path = os.path.join(output_dir, "raw_audio.wav")
    
    # Open the video file
    container = av.open(video_path)
    
    # Get the audio stream
    audio = container.streams.audio[0]
    
    # Initialize resampler to 22050 Hz mono
    resampler = av.AudioResampler(
        format=av.AudioFormat('s16'),
        layout='mono',
        rate=22050
    )
    
    # Read all audio frames and convert to numpy array
    samples = []
    for frame in container.decode(audio):
        frames = resampler.resample(frame)
        for frame in frames:
            array = frame.to_ndarray()
            samples.extend(array.flatten())
    
    # Convert to numpy array and normalize
    samples = np.array(samples, dtype=np.float32)
    samples = samples / np.max(np.abs(samples))  # Normalize
    
    # Save as WAV
    sf.write(output_path, samples, 22050)
    
    # Close the container
    container.close()
    
    return output_path

def segment_audio(audio_path, output_dir, min_segment_length=3.0, max_segment_length=10.0):
    """Split audio into segments based on silence detection."""
    print(f"Segmenting audio from {audio_path}")
    
    # Create segments directory
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=22050)
    
    # Detect speech segments using energy-based voice activity detection
    intervals = librosa.effects.split(
        audio,
        top_db=30,  # Adjust this value to change sensitivity
        frame_length=2048,
        hop_length=512
    )
    
    # Process segments
    segments = []
    for i, (start, end) in enumerate(intervals):
        # Convert frame indices to seconds
        start_time = start / sr
        end_time = end / sr
        duration = end_time - start_time
        
        # Skip segments that are too short or too long
        if duration < min_segment_length or duration > max_segment_length:
            continue
        
        # Extract segment
        segment = audio[start:end]
        
        # Save segment
        segment_path = os.path.join(segments_dir, f"segment_{i:04d}.wav")
        sf.write(segment_path, segment, sr)
        segments.append(segment_path)
        
        print(f"Saved segment {i+1}: {segment_path} ({duration:.2f}s)")
    
    return segments

def main():
    # Setup paths
    video_path = r"C:\Users\Dell\OneDrive\Desktop\video\h9.mp4"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "training_data")
    
    # Extract audio
    audio_path = extract_audio_from_video(video_path, output_dir)
    
    # Segment audio
    segments = segment_audio(audio_path, output_dir)
    
    print(f"\nProcessing complete!")
    print(f"Total segments created: {len(segments)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()