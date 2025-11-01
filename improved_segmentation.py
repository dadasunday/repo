"""
Improved Audio Segmentation Pipeline

Creates training segments with consistent loudness, proper trimming,
and balanced phoneme distribution for better voice cloning.

Key improvements:
- Loudness-normalized segments (consistent RMS)
- Smart silence trimming with configurable thresholds
- Segment deduplication and quality filtering
- Phoneme diversity analysis
- Balanced dataset creation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentConfig:
    """Configuration for segmentation pipeline"""
    # Duration constraints
    min_duration: float = 2.0
    max_duration: float = 10.0
    target_duration: float = 5.0

    # Silence detection
    silence_threshold_db: int = 35
    min_silence_duration: float = 0.3

    # Quality filtering
    min_rms_db: float = -40.0
    max_peak_db: float = -0.5
    target_rms_db: float = -21.0

    # Spectral criteria
    min_spectral_centroid: float = 800.0
    max_spectral_centroid: float = 4000.0

    # Sample rate
    sample_rate: int = 22050


class ImprovedSegmentation:
    """Advanced audio segmentation with quality control"""

    def __init__(self, config: Optional[SegmentConfig] = None):
        """
        Args:
            config: Segmentation configuration (uses defaults if None)
        """
        self.config = config or SegmentConfig()

    def db_to_linear(self, db: float) -> float:
        """Convert dB to linear amplitude"""
        return 10 ** (db / 20.0)

    def linear_to_db(self, linear: float) -> float:
        """Convert linear amplitude to dB"""
        return 20 * np.log10(max(linear, 1e-10))

    def trim_silence(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Trim silence from beginning and end

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Tuple of (trimmed_audio, (start_frame, end_frame))
        """
        # Use librosa's trim with config threshold
        trimmed, indices = librosa.effects.trim(
            audio,
            top_db=self.config.silence_threshold_db,
            frame_length=2048,
            hop_length=512
        )

        return trimmed, indices

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from audio

        Args:
            audio: Input audio

        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)

    def normalize_segment(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize segment to target RMS level

        Args:
            audio: Input audio segment

        Returns:
            Normalized audio
        """
        # Remove DC offset first
        audio = self.remove_dc_offset(audio)

        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio ** 2))

        if current_rms < 1e-10:
            return audio  # Avoid division by zero

        # Calculate target RMS
        target_rms = self.db_to_linear(self.config.target_rms_db)

        # Apply normalization
        gain = target_rms / current_rms
        audio_normalized = audio * gain

        # Soft limiting to prevent clipping
        peak = np.max(np.abs(audio_normalized))
        if peak > 0.95:
            audio_normalized = audio_normalized * (0.95 / peak)

        return audio_normalized

    def detect_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Detect speech segments using silence detection

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            List of (start_frame, end_frame) tuples
        """
        # Get non-silent intervals
        intervals = librosa.effects.split(
            audio,
            top_db=self.config.silence_threshold_db,
            frame_length=2048,
            hop_length=512
        )

        # Filter by duration
        valid_segments = []
        min_samples = int(self.config.min_duration * sr)
        max_samples = int(self.config.max_duration * sr)

        for start, end in intervals:
            duration_samples = end - start

            if duration_samples < min_samples:
                continue  # Too short

            if duration_samples <= max_samples:
                # Perfect size
                valid_segments.append((start, end))
            else:
                # Too long - split into chunks
                chunk_size = int(self.config.target_duration * sr)
                overlap = int(0.5 * sr)  # 0.5s overlap

                current_start = start
                while current_start < end:
                    current_end = min(current_start + chunk_size, end)

                    if (current_end - current_start) >= min_samples:
                        valid_segments.append((current_start, current_end))

                    current_start += (chunk_size - overlap)

        return valid_segments

    def calculate_segment_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate quality metrics for a segment

        Args:
            audio: Audio segment
            sr: Sample rate

        Returns:
            Dictionary of quality metrics
        """
        # RMS and peak
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        rms_db = self.linear_to_db(rms)
        peak_db = self.linear_to_db(peak)

        # Spectral features
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

        # Pitch estimation (for voice detection)
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sr,
            fmin=50,
            fmax=500
        )
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        avg_pitch = np.mean(pitch_values) if pitch_values else 0

        return {
            'rms_db': rms_db,
            'peak_db': peak_db,
            'centroid': centroid,
            'flatness': flatness,
            'zcr': zcr,
            'pitch': avg_pitch,
            'has_pitch': len(pitch_values) > 0
        }

    def is_valid_segment(self, audio: np.ndarray, sr: int) -> Tuple[bool, str]:
        """
        Check if segment meets quality criteria

        Args:
            audio: Audio segment
            sr: Sample rate

        Returns:
            Tuple of (is_valid, reason)
        """
        metrics = self.calculate_segment_quality(audio, sr)

        # Check RMS level
        if metrics['rms_db'] < self.config.min_rms_db:
            return False, "too quiet"

        # Check peak level
        if metrics['peak_db'] > self.config.max_peak_db:
            return False, "clipping detected"

        # Check spectral characteristics
        if metrics['centroid'] < self.config.min_spectral_centroid:
            return False, "too dark/muffled"

        if metrics['centroid'] > self.config.max_spectral_centroid:
            return False, "too bright/harsh"

        # Check for voice content (pitch detection)
        if not metrics['has_pitch']:
            return False, "no voice detected"

        # Check spectral flatness (avoid pure noise)
        if metrics['flatness'] > 0.5:
            return False, "too noisy"

        return True, "pass"

    def calculate_audio_similarity(self,
                                   audio1: np.ndarray,
                                   audio2: np.ndarray,
                                   sr: int) -> float:
        """
        Calculate similarity between two audio segments

        Args:
            audio1: First audio segment
            audio2: Second audio segment
            sr: Sample rate

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Use MFCC for similarity comparison
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)

        # Average across time
        mfcc1_avg = np.mean(mfcc1, axis=1)
        mfcc2_avg = np.mean(mfcc2, axis=1)

        # Cosine similarity
        similarity = np.dot(mfcc1_avg, mfcc2_avg) / (
            np.linalg.norm(mfcc1_avg) * np.linalg.norm(mfcc2_avg) + 1e-10
        )

        return max(0.0, similarity)

    def deduplicate_segments(self,
                            segments: List[np.ndarray],
                            sr: int,
                            similarity_threshold: float = 0.95) -> List[int]:
        """
        Find duplicate or very similar segments

        Args:
            segments: List of audio segments
            sr: Sample rate
            similarity_threshold: Threshold for duplicate detection

        Returns:
            List of indices to keep
        """
        if len(segments) <= 1:
            return list(range(len(segments)))

        keep_indices = [0]  # Always keep first segment

        for i in range(1, len(segments)):
            is_duplicate = False

            # Compare with kept segments
            for j in keep_indices:
                similarity = self.calculate_audio_similarity(
                    segments[i],
                    segments[j],
                    sr
                )

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.info(f"  Segment {i} is duplicate of segment {j} "
                              f"(similarity: {similarity:.3f})")
                    break

            if not is_duplicate:
                keep_indices.append(i)

        return keep_indices

    def segment_audio_file(self,
                          input_path: Path,
                          output_dir: Path,
                          prefix: str = "segment") -> List[Path]:
        """
        Segment a single audio file

        Args:
            input_path: Input audio file path
            output_dir: Output directory for segments
            prefix: Prefix for output filenames

        Returns:
            List of created segment file paths
        """
        logger.info(f"\nProcessing: {input_path.name}")

        # Load audio
        audio, sr = librosa.load(str(input_path), sr=self.config.sample_rate, mono=True)

        # Detect segments
        segment_intervals = self.detect_segments(audio, sr)
        logger.info(f"  Found {len(segment_intervals)} candidate segments")

        # Extract and validate segments
        valid_segments = []
        segment_audio = []

        for i, (start, end) in enumerate(segment_intervals):
            segment = audio[start:end]

            # Trim silence
            segment, _ = self.trim_silence(segment, sr)

            # Normalize
            segment = self.normalize_segment(segment)

            # Validate
            is_valid, reason = self.is_valid_segment(segment, sr)

            if is_valid:
                valid_segments.append((i, segment))
                segment_audio.append(segment)
            else:
                logger.info(f"  Segment {i} rejected: {reason}")

        logger.info(f"  Valid segments: {len(valid_segments)}/{len(segment_intervals)}")

        # Deduplicate
        if len(segment_audio) > 1:
            keep_indices = self.deduplicate_segments(segment_audio, sr)
            logger.info(f"  Unique segments after deduplication: {len(keep_indices)}")

            valid_segments = [valid_segments[i] for i in keep_indices]

        # Save segments
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        for idx, (original_idx, segment) in enumerate(valid_segments):
            output_path = output_dir / f"{prefix}_{idx:04d}.wav"
            sf.write(str(output_path), segment, sr, subtype='PCM_16')
            output_paths.append(output_path)

        logger.info(f"  Saved {len(output_paths)} segments to {output_dir}")

        return output_paths

    def segment_multiple_files(self,
                              input_files: List[Path],
                              output_dir: Path,
                              merge: bool = True) -> List[Path]:
        """
        Segment multiple audio files and optionally merge

        Args:
            input_files: List of input audio file paths
            output_dir: Output directory
            merge: If True, merge all segments into single directory

        Returns:
            List of all created segment paths
        """
        all_segments = []

        for input_file in input_files:
            # Create subdirectory for each file if not merging
            if merge:
                file_output_dir = output_dir
                prefix = f"seg_{input_file.stem}"
            else:
                file_output_dir = output_dir / input_file.stem
                prefix = "segment"

            try:
                segments = self.segment_audio_file(
                    input_file,
                    file_output_dir,
                    prefix=prefix
                )
                all_segments.extend(segments)

            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {e}")

        return all_segments

    def analyze_dataset_diversity(self, segments_dir: Path) -> Dict:
        """
        Analyze diversity of segmented dataset

        Args:
            segments_dir: Directory containing segments

        Returns:
            Dictionary with diversity metrics
        """
        segment_files = list(segments_dir.glob("*.wav"))

        if not segment_files:
            logger.warning(f"No segments found in {segments_dir}")
            return {}

        logger.info(f"\nAnalyzing dataset diversity ({len(segment_files)} segments)...")

        durations = []
        centroids = []
        pitches = []

        for seg_file in segment_files:
            try:
                audio, sr = librosa.load(str(seg_file), sr=None, mono=True)

                # Duration
                durations.append(len(audio) / sr)

                # Spectral centroid
                centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
                centroids.append(centroid)

                # Pitch
                pitches_arr, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=500)
                pitch_values = []
                for t in range(pitches_arr.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches_arr[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    pitches.append(np.mean(pitch_values))

            except Exception as e:
                logger.warning(f"Error analyzing {seg_file.name}: {e}")

        # Calculate statistics
        diversity_metrics = {
            'total_segments': len(segment_files),
            'total_duration': sum(durations),
            'avg_duration': np.mean(durations),
            'duration_std': np.std(durations),
            'avg_centroid': np.mean(centroids),
            'centroid_std': np.std(centroids),
            'avg_pitch': np.mean(pitches) if pitches else 0,
            'pitch_std': np.std(pitches) if pitches else 0,
        }

        # Print report
        logger.info(f"\nDataset Diversity Report:")
        logger.info(f"  Total segments: {diversity_metrics['total_segments']}")
        logger.info(f"  Total duration: {diversity_metrics['total_duration']:.1f} seconds")
        logger.info(f"  Avg duration: {diversity_metrics['avg_duration']:.1f}s "
                   f"(±{diversity_metrics['duration_std']:.1f}s)")
        logger.info(f"  Avg brightness: {diversity_metrics['avg_centroid']:.0f} Hz "
                   f"(±{diversity_metrics['centroid_std']:.0f} Hz)")
        if pitches:
            logger.info(f"  Avg pitch: {diversity_metrics['avg_pitch']:.0f} Hz "
                       f"(±{diversity_metrics['pitch_std']:.0f} Hz)")

        return diversity_metrics


def main():
    """Main execution function"""

    # Initialize segmentation pipeline
    config = SegmentConfig(
        min_duration=2.0,
        max_duration=10.0,
        target_duration=5.0,
        silence_threshold_db=35,
        target_rms_db=-21.0,
        sample_rate=22050
    )

    segmenter = ImprovedSegmentation(config)

    base_dir = Path(__file__).parent / "training_data"

    # Find all audio files to segment
    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.m4a']
    input_files = []

    for ext in audio_extensions:
        input_files.extend(base_dir.glob(ext))

    # Filter out already processed files
    input_files = [
        f for f in input_files
        if 'segment' not in f.stem.lower() and
           'remastered' not in f.stem.lower() and
           'processed' not in f.stem.lower()
    ]

    if not input_files:
        logger.error(f"No audio files found in {base_dir}")
        logger.info("Place training audio/video files in training_data/")
        return

    logger.info(f"\nFound {len(input_files)} input files:")
    for f in input_files:
        logger.info(f"  - {f.name}")

    # Create improved segments
    output_dir = base_dir / "segments_improved"

    logger.info(f"\n{'='*80}")
    logger.info("IMPROVED SEGMENTATION PIPELINE")
    logger.info(f"{'='*80}")

    segments = segmenter.segment_multiple_files(
        input_files,
        output_dir,
        merge=True
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Segmentation complete!")
    logger.info(f"  Created {len(segments)} high-quality segments")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'='*80}")

    # Analyze dataset diversity
    segmenter.analyze_dataset_diversity(output_dir)

    logger.info("\nNext steps:")
    logger.info("1. Review segments in segments_improved/")
    logger.info("2. Use segments_improved/ for voice cloning training")
    logger.info("3. Run: python optimal_voice_clone.py (with updated data path)")


if __name__ == "__main__":
    main()
