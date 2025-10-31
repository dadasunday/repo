"""
Process all new training videos and merge with existing data
"""

from pathlib import Path
import subprocess
import imageio_ffmpeg
import librosa
import soundfile as sf
import numpy as np
import shutil

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file."""
    print(f"  Extracting: {video_path.name}")

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_path,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '22050',
            '-ac', '1',
            '-y',
            str(output_audio_path)
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        audio, sr = librosa.load(str(output_audio_path), sr=None)
        duration = len(audio) / sr
        print(f"    [OK] Duration: {duration:.1f}s")
        return True

    except Exception as e:
        print(f"    [ERROR] {e}")
        return False

def segment_audio(audio_path, output_dir, prefix, min_duration=2.0, max_duration=10.0):
    """Segment audio into training clips."""

    audio, sr = librosa.load(str(audio_path), sr=None)
    intervals = librosa.effects.split(audio, top_db=30)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for start, end in intervals:
        segment = audio[start:end]
        duration_sec = len(segment) / sr

        if min_duration <= duration_sec <= max_duration:
            rms = np.sqrt(np.mean(segment**2))

            if rms > 0.02:
                output_file = output_dir / f"{prefix}_{saved_count:04d}.wav"
                sf.write(str(output_file), segment, sr)
                saved_count += 1

    return saved_count

def main():
    print("="*70)
    print("PROCESS ALL NEW TRAINING DATA")
    print("="*70)

    base_path = Path(__file__).parent
    training_data_path = base_path / "training_data"

    # Find all video files
    video_files = sorted(training_data_path.glob("*.mp4"))

    if not video_files:
        print("[ERROR] No video files found")
        return

    print(f"\nFound {len(video_files)} video files")

    # Process each video
    print("\n[STEP 1/3] Extracting audio from all videos...")

    all_audio_files = []

    for i, video_path in enumerate(video_files, 1):
        audio_output = training_data_path / f"extracted_audio_{i}.wav"

        if extract_audio_from_video(video_path, audio_output):
            all_audio_files.append(audio_output)

    print(f"\n[OK] Extracted {len(all_audio_files)} audio files")

    # Segment all audio files
    print("\n[STEP 2/3] Segmenting all audio files...")

    all_segments_dir = training_data_path / "segments_all"
    all_segments_dir.mkdir(parents=True, exist_ok=True)

    total_segments = 0

    for i, audio_file in enumerate(all_audio_files, 1):
        print(f"  [{i}/{len(all_audio_files)}] {audio_file.name}")
        count = segment_audio(audio_file, all_segments_dir, f"video{i}", min_duration=2.0, max_duration=10.0)
        print(f"    Saved {count} segments")
        total_segments += count

    print(f"\n[OK] Total segments: {total_segments}")

    # Merge with old segments
    print("\n[STEP 3/3] Creating final merged dataset...")

    old_segments_dir = training_data_path / "segments"
    final_merged_dir = training_data_path / "segments_final_merged"
    final_merged_dir.mkdir(parents=True, exist_ok=True)

    # Copy old segments
    old_segments = list(old_segments_dir.glob("*.wav"))
    for seg in old_segments:
        shutil.copy2(seg, final_merged_dir / f"original_{seg.name}")

    # Copy new segments
    new_segments = list(all_segments_dir.glob("*.wav"))
    for seg in new_segments:
        shutil.copy2(seg, final_merged_dir / f"new_{seg.name}")

    final_count = len(list(final_merged_dir.glob("*.wav")))

    print(f"\n[OK] Final merged dataset: {final_count} segments")
    print(f"    Old segments: {len(old_segments)}")
    print(f"    New segments: {len(new_segments)}")

    # Analyze quality
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)

    total_duration = 0
    for seg in final_merged_dir.glob("*.wav"):
        audio, sr = librosa.load(str(seg), sr=None)
        total_duration += len(audio) / sr

    print(f"\nTotal segments: {final_count}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Average segment: {total_duration/final_count:.1f}s")
    print(f"\nFinal dataset location: {final_merged_dir}")

    print("\n" + "="*70)
    print("NEXT STEP: Update improved_voice_clone.py to use:")
    print(f"  data_path = base_path / 'training_data' / 'segments_final_merged'")
    print("="*70)

if __name__ == "__main__":
    main()
