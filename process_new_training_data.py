"""
Process New Training Data from Video
Extracts audio from video, segments it, and prepares for voice cloning
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import subprocess
import imageio_ffmpeg

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file using ffmpeg."""
    print(f"Extracting audio from: {video_path.name}")

    try:
        # Get ffmpeg binary path from imageio-ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"  Using ffmpeg from: {ffmpeg_path}")

        # Extract audio using ffmpeg
        cmd = [
            ffmpeg_path,
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '22050',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            str(output_audio_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 or output_audio_path.exists():
            # Verify the output
            audio, sr = librosa.load(str(output_audio_path), sr=None)
            print(f"[OK] Audio extracted to: {output_audio_path.name}")
            print(f"     Duration: {len(audio)/sr:.1f}s, Sample rate: {sr}Hz")
            return True
        else:
            print(f"[ERROR] ffmpeg failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to extract audio: {e}")
        print("\nAlternative: Please manually convert the video to audio:")
        print(f"  1. Use an online converter (e.g., cloudconvert.com)")
        print(f"  2. Or use VLC: Media > Convert/Save > Convert to Audio")
        print(f"  3. Save as WAV file in training_data/ folder")
        return False

def analyze_audio_quality(audio_path):
    """Analyze audio file quality."""
    print(f"\nAnalyzing: {audio_path.name}")

    audio, sr = librosa.load(str(audio_path), sr=None)
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))

    # Pitch analysis
    pitches, mags = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=500)
    pitch_values = [pitches[mags[:, t].argmax(), t] for t in range(pitches.shape[1])
                    if pitches[mags[:, t].argmax(), t] > 0]
    avg_pitch = np.mean(pitch_values) if pitch_values else 0

    # Spectral analysis
    brightness = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])

    # Silence analysis
    intervals = librosa.effects.split(audio, top_db=30)
    speech_duration = sum([(end - start) / sr for start, end in intervals])
    speech_ratio = speech_duration / duration if duration > 0 else 0

    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  RMS Energy: {rms:.4f}")
    print(f"  Avg Pitch: {avg_pitch:.1f} Hz")
    print(f"  Brightness: {brightness:.0f} Hz")
    print(f"  Speech Ratio: {speech_ratio*100:.1f}%")

    return {
        'duration': duration,
        'sr': sr,
        'rms': rms,
        'pitch': avg_pitch,
        'brightness': brightness,
        'speech_ratio': speech_ratio
    }

def segment_audio(audio_path, output_dir, min_segment_duration=2.0, max_segment_duration=10.0):
    """Segment audio into training clips based on silence."""
    print(f"\nSegmenting audio: {audio_path.name}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None)

    # Split on silence using librosa
    print("  Detecting speech segments...")
    intervals = librosa.effects.split(audio, top_db=30)

    print(f"  Found {len(intervals)} initial segments")

    # Filter and save segments
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for start, end in intervals:
        # Extract segment
        segment = audio[start:end]
        duration_sec = len(segment) / sr

        # Filter by duration
        if min_segment_duration <= duration_sec <= max_segment_duration:
            # Check audio quality
            rms = np.sqrt(np.mean(segment**2))

            # Filter by RMS (avoid very quiet segments)
            if rms > 0.02:
                output_file = output_dir / f"segment_new_{saved_count:04d}.wav"
                sf.write(str(output_file), segment, sr)
                saved_count += 1

    print(f"[OK] Saved {saved_count} quality segments to: {output_dir}")
    return saved_count

def merge_training_datasets(old_segments_dir, new_segments_dir, merged_dir):
    """Merge old and new training segments."""
    print(f"\nMerging training datasets...")

    merged_dir.mkdir(parents=True, exist_ok=True)

    # Copy old segments
    old_segments = list(old_segments_dir.glob("*.wav"))
    print(f"  Old segments: {len(old_segments)}")

    for i, seg in enumerate(old_segments):
        import shutil
        shutil.copy2(seg, merged_dir / f"merged_{i:04d}_old.wav")

    # Copy new segments
    new_segments = list(new_segments_dir.glob("*.wav"))
    print(f"  New segments: {len(new_segments)}")

    for i, seg in enumerate(new_segments):
        import shutil
        shutil.copy2(seg, merged_dir / f"merged_{i:04d}_new.wav")

    total = len(old_segments) + len(new_segments)
    print(f"[OK] Merged dataset: {total} total segments")
    return total

def main():
    print("="*70)
    print("PROCESS NEW TRAINING DATA FROM VIDEO")
    print("="*70)

    # Setup paths
    base_path = Path(__file__).parent
    training_data_path = base_path / "training_data"

    # Find video file
    video_files = list(training_data_path.glob("*.mp4")) + list(training_data_path.glob("*.mov"))

    if not video_files:
        print("[ERROR] No video files found in training_data/")
        return

    video_path = video_files[0]
    print(f"\nFound video: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / (1024*1024):.1f} MB")

    # Step 1: Extract audio from video
    print("\n" + "="*70)
    print("[STEP 1/5] Extract Audio from Video")
    print("="*70)

    new_audio_path = training_data_path / "new_training_audio.wav"

    if not extract_audio_from_video(video_path, new_audio_path):
        print("[ERROR] Failed to extract audio. Aborting.")
        return

    # Step 2: Analyze audio quality
    print("\n" + "="*70)
    print("[STEP 2/5] Analyze Audio Quality")
    print("="*70)

    audio_stats = analyze_audio_quality(new_audio_path)

    # Step 3: Segment audio
    print("\n" + "="*70)
    print("[STEP 3/5] Segment Audio")
    print("="*70)

    new_segments_dir = training_data_path / "segments_new"
    num_segments = segment_audio(new_audio_path, new_segments_dir)

    if num_segments == 0:
        print("[WARNING] No quality segments found. Check audio quality.")
        return

    # Step 4: Merge with existing data
    print("\n" + "="*70)
    print("[STEP 4/5] Merge Training Datasets")
    print("="*70)

    old_segments_dir = training_data_path / "segments"
    merged_segments_dir = training_data_path / "segments_merged"

    total_segments = merge_training_datasets(old_segments_dir, new_segments_dir, merged_segments_dir)

    # Step 5: Update voice cloning script
    print("\n" + "="*70)
    print("[STEP 5/5] Ready for Voice Cloning")
    print("="*70)

    print("\n[OK] Processing Complete!")
    print(f"\nTraining data summary:")
    print(f"  Old dataset: {len(list(old_segments_dir.glob('*.wav')))} segments")
    print(f"  New dataset: {num_segments} segments")
    print(f"  Merged dataset: {total_segments} segments")
    print(f"\nNext steps:")
    print(f"  1. Review segments in: {merged_segments_dir}")
    print(f"  2. Update improved_voice_clone.py to use merged segments")
    print(f"  3. Run: python improved_voice_clone.py")

    # Show command to update
    print(f"\n" + "="*70)
    print("TO USE MERGED DATASET:")
    print("="*70)
    print("\nEdit improved_voice_clone.py, line 208:")
    print("  OLD: data_path = base_path / 'training_data' / 'segments'")
    print("  NEW: data_path = base_path / 'training_data' / 'segments_merged'")
    print("\nOr run this command:")
    print("  sed -i 's/segments/segments_merged/g' improved_voice_clone.py")
    print("="*70)

if __name__ == "__main__":
    main()
