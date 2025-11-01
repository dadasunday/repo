"""
Training Data Remastering Script

Fixes clipped peaks, normalizes loudness, and improves spectral consistency
in training data to match production audio characteristics.

Key improvements:
- Removes 0 dBFS clipping using true-peak limiting
- Normalizes RMS to consistent levels (−21 to −18 dBFS)
- Controls crest factor to realistic range (7-9 dB)
- Applies gentle de-essing to reduce excessive brightness
- Preserves natural dynamics while improving consistency
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataRemaster:
    """Remaster training audio to remove artifacts and normalize characteristics"""

    def __init__(self,
                 target_rms_db: float = -21.0,
                 target_peak_db: float = -3.0,
                 target_crest_db: float = 8.0,
                 sr: int = 22050):
        """
        Args:
            target_rms_db: Target RMS level in dBFS (default -21)
            target_peak_db: Target peak level in dBFS (default -3 for headroom)
            target_crest_db: Target crest factor in dB (default 8)
            sr: Sample rate
        """
        self.target_rms_db = target_rms_db
        self.target_peak_db = target_peak_db
        self.target_crest_db = target_crest_db
        self.sr = sr

    def db_to_linear(self, db: float) -> float:
        """Convert dB to linear amplitude"""
        return 10 ** (db / 20.0)

    def linear_to_db(self, linear: float) -> float:
        """Convert linear amplitude to dB"""
        return 20 * np.log10(max(linear, 1e-10))

    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS value"""
        return np.sqrt(np.mean(audio ** 2))

    def calculate_peak(self, audio: np.ndarray) -> float:
        """Calculate peak value"""
        return np.max(np.abs(audio))

    def calculate_crest_factor(self, audio: np.ndarray) -> float:
        """Calculate crest factor (peak-to-RMS ratio) in dB"""
        peak = self.calculate_peak(audio)
        rms = self.calculate_rms(audio)
        return self.linear_to_db(peak / rms) if rms > 0 else 0.0

    def apply_true_peak_limiter(self,
                                audio: np.ndarray,
                                threshold_db: float = -1.0,
                                release_samples: int = 220) -> np.ndarray:
        """
        Apply true-peak limiting to prevent clipping

        Args:
            audio: Input audio
            threshold_db: Limiting threshold in dBFS
            release_samples: Release time in samples (220 = 10ms at 22050Hz)
        """
        threshold = self.db_to_linear(threshold_db)

        # Create gain envelope
        peak_env = np.abs(audio)
        gain = np.ones_like(audio)

        # Apply limiting where peaks exceed threshold
        for i in range(len(audio)):
            if peak_env[i] > threshold:
                gain[i] = threshold / peak_env[i]
            elif i > 0:
                # Smooth release
                gain[i] = min(1.0, gain[i-1] + (1.0 / release_samples))

        # Apply minimal smoothing to prevent artifacts
        window = signal.windows.hann(11)
        window /= window.sum()
        gain_smooth = signal.convolve(gain, window, mode='same')

        return audio * gain_smooth

    def remove_clipped_peaks(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect and reconstruct clipped peaks using interpolation

        Args:
            audio: Input audio with potential clipping
        """
        # Detect clipping (samples at or near full scale)
        clipping_threshold = 0.99
        clipped = np.abs(audio) >= clipping_threshold

        if not np.any(clipped):
            return audio  # No clipping detected

        logger.info(f"  Detected {np.sum(clipped)} clipped samples ({100*np.mean(clipped):.2f}%)")

        # Find clipped regions
        clipped_indices = np.where(clipped)[0]

        # Reconstruct using cubic interpolation
        audio_fixed = audio.copy()
        non_clipped_indices = np.where(~clipped)[0]

        if len(non_clipped_indices) > 4:  # Need enough points for cubic interpolation
            from scipy.interpolate import interp1d
            interpolator = interp1d(
                non_clipped_indices,
                audio[non_clipped_indices],
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            audio_fixed[clipped] = interpolator(clipped_indices)

        return audio_fixed

    def apply_de_esser(self,
                       audio: np.ndarray,
                       freq_range: Tuple[float, float] = (4000, 10000),
                       reduction_db: float = 3.0) -> np.ndarray:
        """
        Apply gentle de-essing to reduce excessive sibilance/brightness

        Args:
            audio: Input audio
            freq_range: Frequency range to target (Hz)
            reduction_db: Amount of reduction in dB
        """
        # Design bandpass filter for sibilance range
        nyquist = self.sr / 2
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist

        sos = signal.butter(4, [low, high], btype='band', output='sos')
        sibilance = signal.sosfilt(sos, audio)

        # Calculate envelope of sibilance
        sibilance_env = np.abs(sibilance)

        # Smooth envelope
        window_size = int(self.sr * 0.01)  # 10ms window
        kernel = np.ones(window_size) / window_size
        sibilance_env = np.convolve(sibilance_env, kernel, mode='same')

        # Calculate dynamic gain reduction
        threshold = np.percentile(sibilance_env, 70)  # Dynamic threshold
        reduction_linear = self.db_to_linear(-reduction_db)

        gain = np.ones_like(audio)
        over_threshold = sibilance_env > threshold
        gain[over_threshold] = 1.0 - (1.0 - reduction_linear) * \
                                (sibilance_env[over_threshold] - threshold) / \
                                (np.max(sibilance_env) - threshold + 1e-10)

        # Apply only to high frequencies
        high_freq = signal.sosfilt(sos, audio)
        high_freq_reduced = high_freq * gain

        # Reconstruct: original low/mid + reduced high
        audio_deessed = audio - high_freq + high_freq_reduced

        return audio_deessed

    def normalize_to_target_rms(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target RMS level

        Args:
            audio: Input audio
        """
        current_rms = self.calculate_rms(audio)
        target_rms = self.db_to_linear(self.target_rms_db)

        if current_rms > 0:
            gain = target_rms / current_rms
            audio = audio * gain

        return audio

    def control_crest_factor(self,
                            audio: np.ndarray,
                            attack_samples: int = 11,
                            release_samples: int = 220) -> np.ndarray:
        """
        Apply gentle compression to control crest factor

        Args:
            audio: Input audio
            attack_samples: Attack time in samples
            release_samples: Release time in samples
        """
        current_crest = self.calculate_crest_factor(audio)

        if current_crest <= self.target_crest_db:
            return audio  # Already within target

        # Calculate threshold for compression
        rms = self.calculate_rms(audio)
        target_peak = rms * self.db_to_linear(self.target_crest_db)
        threshold = target_peak * 0.7  # Start compressing before target peak

        # Apply gentle compression
        envelope = np.abs(audio)
        gain = np.ones_like(audio)
        gain_state = 1.0

        for i in range(len(audio)):
            if envelope[i] > threshold:
                # Soft-knee compression ratio (2:1)
                target_gain = threshold / envelope[i]
                target_gain = target_gain ** 0.5  # Soften compression

                # Attack
                if target_gain < gain_state:
                    gain_state = gain_state * 0.8 + target_gain * 0.2
                # Release
                else:
                    gain_state = min(1.0, gain_state + (1.0 / release_samples))
            else:
                gain_state = min(1.0, gain_state + (1.0 / release_samples))

            gain[i] = gain_state

        return audio * gain

    def remaster_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply full remastering pipeline

        Args:
            audio: Input audio

        Returns:
            Tuple of (remastered_audio, metrics_dict)
        """
        # Calculate original metrics
        original_metrics = {
            'rms_db': self.linear_to_db(self.calculate_rms(audio)),
            'peak_db': self.linear_to_db(self.calculate_peak(audio)),
            'crest_db': self.calculate_crest_factor(audio)
        }

        logger.info(f"  Original: RMS={original_metrics['rms_db']:.1f}dB "
                   f"Peak={original_metrics['peak_db']:.1f}dB "
                   f"Crest={original_metrics['crest_db']:.1f}dB")

        # Step 1: Remove clipped peaks
        audio = self.remove_clipped_peaks(audio)

        # Step 2: Apply gentle de-essing
        audio = self.apply_de_esser(audio, reduction_db=2.5)

        # Step 3: Normalize to target RMS
        audio = self.normalize_to_target_rms(audio)

        # Step 4: Control crest factor
        audio = self.control_crest_factor(audio)

        # Step 5: Final true-peak limiting
        audio = self.apply_true_peak_limiter(audio, threshold_db=self.target_peak_db)

        # Calculate final metrics
        final_metrics = {
            'rms_db': self.linear_to_db(self.calculate_rms(audio)),
            'peak_db': self.linear_to_db(self.calculate_peak(audio)),
            'crest_db': self.calculate_crest_factor(audio)
        }

        logger.info(f"  Remastered: RMS={final_metrics['rms_db']:.1f}dB "
                   f"Peak={final_metrics['peak_db']:.1f}dB "
                   f"Crest={final_metrics['crest_db']:.1f}dB")

        return audio, {'original': original_metrics, 'remastered': final_metrics}

    def remaster_file(self,
                     input_path: Path,
                     output_path: Path) -> Dict:
        """
        Remaster a single audio file

        Args:
            input_path: Path to input WAV file
            output_path: Path to output WAV file

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Processing: {input_path.name}")

        # Load audio
        audio, sr = librosa.load(str(input_path), sr=self.sr, mono=True)

        # Remaster
        audio_remastered, metrics = self.remaster_audio(audio)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save remastered audio
        sf.write(str(output_path), audio_remastered, self.sr, subtype='PCM_16')

        return metrics

    def remaster_directory(self,
                          input_dir: Path,
                          output_dir: Path,
                          pattern: str = "*.wav") -> None:
        """
        Remaster all audio files in a directory

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: File pattern to match (default "*.wav")
        """
        input_files = list(input_dir.glob(pattern))

        if not input_files:
            logger.warning(f"No files matching '{pattern}' found in {input_dir}")
            return

        logger.info(f"\nRemastering {len(input_files)} files from {input_dir}")
        logger.info(f"Output directory: {output_dir}\n")

        output_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = []

        for input_file in input_files:
            output_file = output_dir / input_file.name

            try:
                metrics = self.remaster_file(input_file, output_file)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {e}")

        # Print summary statistics
        if all_metrics:
            logger.info("\n" + "="*60)
            logger.info("REMASTERING SUMMARY")
            logger.info("="*60)

            orig_rms = np.mean([m['original']['rms_db'] for m in all_metrics])
            orig_peak = np.mean([m['original']['peak_db'] for m in all_metrics])
            orig_crest = np.mean([m['original']['crest_db'] for m in all_metrics])

            final_rms = np.mean([m['remastered']['rms_db'] for m in all_metrics])
            final_peak = np.mean([m['remastered']['peak_db'] for m in all_metrics])
            final_crest = np.mean([m['remastered']['crest_db'] for m in all_metrics])

            logger.info(f"Files processed: {len(all_metrics)}")
            logger.info(f"\nOriginal average:")
            logger.info(f"  RMS: {orig_rms:.1f} dBFS")
            logger.info(f"  Peak: {orig_peak:.1f} dBFS")
            logger.info(f"  Crest: {orig_crest:.1f} dB")
            logger.info(f"\nRemastered average:")
            logger.info(f"  RMS: {final_rms:.1f} dBFS")
            logger.info(f"  Peak: {final_peak:.1f} dBFS")
            logger.info(f"  Crest: {final_crest:.1f} dB")
            logger.info(f"\nTarget values:")
            logger.info(f"  RMS: {self.target_rms_db:.1f} dBFS")
            logger.info(f"  Peak: {self.target_peak_db:.1f} dBFS")
            logger.info(f"  Crest: {self.target_crest_db:.1f} dB")
            logger.info("="*60 + "\n")


def main():
    """Main execution function"""

    # Initialize remaster processor
    remaster = TrainingDataRemaster(
        target_rms_db=-21.0,    # Match training RMS
        target_peak_db=-3.0,    # Leave headroom (not 0 dBFS!)
        target_crest_db=8.0,    # Realistic crest factor
        sr=22050
    )

    # Define paths
    base_dir = Path(__file__).parent / "training_data"

    # Remaster main training segments
    input_dir = base_dir / "segments_final_merged"
    output_dir = base_dir / "segments_remastered"

    if input_dir.exists():
        logger.info("Remastering training segments...")
        remaster.remaster_directory(input_dir, output_dir)
    else:
        logger.warning(f"Directory not found: {input_dir}")

    # Also remaster raw training audio files
    raw_files = [
        "raw_audio.wav",
        "new_training_audio.wav",
        "extracted_audio_1.wav",
        "extracted_audio_2.wav",
        "extracted_audio_3.wav",
        "extracted_audio_4.wav",
        "extracted_audio_5.wav",
        "extracted_audio_6.wav"
    ]

    logger.info("\nRemastering raw audio files...")
    for filename in raw_files:
        input_file = base_dir / filename
        if input_file.exists():
            output_file = base_dir / f"remastered_{filename}"
            try:
                remaster.remaster_file(input_file, output_file)
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

    logger.info("\n✓ Remastering complete!")
    logger.info(f"\nRemastered segments: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Use segments_remastered/ for training instead of segments_final_merged/")
    logger.info("2. Re-train or regenerate voice samples with cleaned data")
    logger.info("3. Apply matching post-processing to production outputs (see production_post_process.py)")


if __name__ == "__main__":
    main()
