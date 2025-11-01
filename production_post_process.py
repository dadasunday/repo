"""
Production Audio Post-Processing Pipeline

Applies loudness normalization, spectral matching, and quality enhancement
to generated voice samples to match training data characteristics.

Key features:
- LUFS-based loudness normalization (broadcast standard)
- Spectral matching to transfer training brightness
- True-peak limiting for distribution
- Dynamic range control
- Automated quality validation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal, interpolate
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionPostProcessor:
    """Post-process generated audio to match training characteristics"""

    def __init__(self,
                 target_lufs: float = -16.0,
                 target_peak_db: float = -1.0,
                 sr: int = 22050):
        """
        Args:
            target_lufs: Target loudness in LUFS (default -16 for modern delivery)
            target_peak_db: Target true peak in dBFS (default -1 for headroom)
            sr: Sample rate
        """
        self.target_lufs = target_lufs
        self.target_peak_db = target_peak_db
        self.sr = sr

    def db_to_linear(self, db: float) -> float:
        """Convert dB to linear amplitude"""
        return 10 ** (db / 20.0)

    def linear_to_db(self, linear: float) -> float:
        """Convert linear amplitude to dB"""
        return 20 * np.log10(max(linear, 1e-10))

    def calculate_lufs(self, audio: np.ndarray) -> float:
        """
        Calculate integrated LUFS (simplified ITU-R BS.1770 approximation)

        This is a simplified version that gives good results for voice.
        For precise LUFS, use pyloudnorm library.

        Args:
            audio: Input audio

        Returns:
            Integrated LUFS value
        """
        # K-weighting filter approximation for speech
        # Pre-filter (high-pass)
        sos_hp = signal.butter(2, 38.0 / (self.sr/2), btype='high', output='sos')
        audio_filtered = signal.sosfilt(sos_hp, audio)

        # RLB filter (high-frequency shelf)
        # Simplified version of the RLB weighting
        b, a = signal.iirfilter(
            2,
            4000.0 / (self.sr/2),
            btype='high',
            ftype='butter'
        )
        audio_filtered = signal.filtfilt(b, a, audio_filtered)

        # Calculate mean square with gating
        block_size = int(0.4 * self.sr)  # 400ms blocks
        hop_size = int(0.1 * self.sr)    # 100ms hop

        blocks = []
        for i in range(0, len(audio_filtered) - block_size, hop_size):
            block = audio_filtered[i:i+block_size]
            mean_square = np.mean(block ** 2)
            blocks.append(mean_square)

        if not blocks:
            return -70.0  # Very quiet

        blocks = np.array(blocks)

        # Absolute threshold gate at -70 LUFS
        absolute_threshold = self.db_to_linear(-70.0) ** 2
        gated_blocks = blocks[blocks >= absolute_threshold]

        if len(gated_blocks) == 0:
            return -70.0

        # Relative threshold gate
        relative_threshold = self.db_to_linear(-10.0) ** 2 * np.mean(gated_blocks)
        gated_blocks = gated_blocks[gated_blocks >= relative_threshold]

        if len(gated_blocks) == 0:
            return -70.0

        # Calculate integrated loudness
        mean_square = np.mean(gated_blocks)
        lufs = -0.691 + 10 * np.log10(mean_square)

        return lufs

    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target LUFS

        Args:
            audio: Input audio

        Returns:
            Loudness-normalized audio
        """
        current_lufs = self.calculate_lufs(audio)
        gain_db = self.target_lufs - current_lufs
        gain_linear = self.db_to_linear(gain_db)

        logger.info(f"  Loudness: {current_lufs:.1f} LUFS → {self.target_lufs:.1f} LUFS "
                   f"(gain: {gain_db:+.1f} dB)")

        return audio * gain_linear

    def apply_true_peak_limiter(self,
                                audio: np.ndarray,
                                threshold_db: float = -1.0) -> np.ndarray:
        """
        Apply transparent true-peak limiter

        Args:
            audio: Input audio
            threshold_db: Limiting threshold in dBFS

        Returns:
            Limited audio
        """
        threshold = self.db_to_linear(threshold_db)
        peak = np.max(np.abs(audio))

        if peak <= threshold:
            return audio  # No limiting needed

        # Calculate required gain reduction
        gain = threshold / peak

        # Apply soft limiting with look-ahead
        lookahead_samples = int(0.001 * self.sr)  # 1ms look-ahead
        release_samples = int(0.05 * self.sr)     # 50ms release

        envelope = np.abs(audio)

        # Look-ahead: shift envelope forward
        envelope_padded = np.pad(envelope, (lookahead_samples, 0), mode='edge')
        envelope = envelope_padded[:len(audio)]

        # Apply limiting gain
        gain_curve = np.ones_like(audio)
        gain_state = 1.0

        for i in range(len(audio)):
            if envelope[i] > threshold:
                target_gain = threshold / envelope[i]

                # Soft-knee
                if target_gain < gain_state:
                    gain_state = target_gain
                else:
                    # Release
                    gain_state = min(1.0, gain_state + (1.0 / release_samples))
            else:
                gain_state = min(1.0, gain_state + (1.0 / release_samples))

            gain_curve[i] = gain_state

        # Smooth gain curve to prevent artifacts
        window_size = int(0.001 * self.sr)  # 1ms smoothing
        if window_size > 1:
            kernel = np.hanning(window_size)
            kernel /= kernel.sum()
            gain_curve = signal.convolve(gain_curve, kernel, mode='same')

        limited = audio * gain_curve

        logger.info(f"  Peak limiting: {self.linear_to_db(peak):.1f} dBFS → "
                   f"{self.linear_to_db(np.max(np.abs(limited))):.1f} dBFS")

        return limited

    def calculate_spectral_envelope(self,
                                    audio: np.ndarray,
                                    n_bands: int = 32) -> np.ndarray:
        """
        Calculate spectral envelope using FFT

        Args:
            audio: Input audio
            n_bands: Number of frequency bands

        Returns:
            Spectral envelope (in dB per band)
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Average over time
        avg_spectrum = np.mean(magnitude, axis=1)

        # Convert to mel scale for perceptual spacing
        mel_spectrum = librosa.feature.melspectrogram(
            S=magnitude**2,
            sr=self.sr,
            n_mels=n_bands,
            fmin=50,
            fmax=self.sr/2
        )
        mel_envelope = np.mean(mel_spectrum, axis=1)

        # Convert to dB
        mel_envelope_db = librosa.power_to_db(mel_envelope, ref=np.max)

        return mel_envelope_db

    def apply_spectral_matching(self,
                                audio: np.ndarray,
                                reference_audio: np.ndarray,
                                strength: float = 0.7) -> np.ndarray:
        """
        Apply spectral matching to transfer tonal characteristics

        Args:
            audio: Audio to process
            reference_audio: Reference audio with target spectrum
            strength: Matching strength (0=none, 1=full)

        Returns:
            Spectrally matched audio
        """
        # Calculate spectral envelopes
        target_envelope = self.calculate_spectral_envelope(reference_audio, n_bands=32)
        current_envelope = self.calculate_spectral_envelope(audio, n_bands=32)

        # Calculate correction curve
        correction_db = (target_envelope - current_envelope) * strength

        logger.info(f"  Spectral match strength: {strength*100:.0f}% "
                   f"(avg correction: {np.mean(np.abs(correction_db)):.1f} dB)")

        # Apply correction using mel-scaled EQ
        # Convert audio to mel spectrogram
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=2048,
            n_mels=32,
            fmin=50,
            fmax=self.sr/2
        )

        # Convert correction to linear gain
        correction_linear = 10 ** (correction_db / 20.0)

        # Apply correction to each frequency bin
        # Map mel bands back to FFT bins
        correction_fft = np.dot(correction_linear, mel_basis)

        # Reshape for broadcasting
        correction_fft = correction_fft[:, np.newaxis]

        # Apply correction
        magnitude_corrected = magnitude * correction_fft

        # Reconstruct audio
        stft_corrected = magnitude_corrected * np.exp(1j * phase)
        audio_matched = librosa.istft(stft_corrected, hop_length=512, length=len(audio))

        return audio_matched

    def enhance_brightness_targeted(self,
                                   audio: np.ndarray,
                                   target_centroid_hz: float = 2800.0) -> np.ndarray:
        """
        Enhance brightness to match target spectral centroid

        Args:
            audio: Input audio
            target_centroid_hz: Target spectral centroid in Hz

        Returns:
            Brightness-enhanced audio
        """
        # Calculate current centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        current_centroid = np.mean(centroid)

        if current_centroid >= target_centroid_hz:
            logger.info(f"  Brightness: {current_centroid:.0f} Hz (already bright enough)")
            return audio

        # Calculate required boost
        centroid_ratio = target_centroid_hz / current_centroid
        boost_db = min(6.0, 3.0 * (centroid_ratio - 1.0))  # Cap at +6dB

        logger.info(f"  Brightness: {current_centroid:.0f} Hz → {target_centroid_hz:.0f} Hz "
                   f"(boost: +{boost_db:.1f} dB)")

        # Apply high-shelf filter
        nyquist = self.sr / 2
        cutoff = 2000 / nyquist

        sos = signal.butter(2, cutoff, btype='high', output='sos')
        audio_high = signal.sosfilt(sos, audio)

        # Mix with original
        boost_linear = self.db_to_linear(boost_db)
        audio_bright = audio + audio_high * (boost_linear - 1.0) * 0.4

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio_bright))
        if peak > 1.0:
            audio_bright /= peak

        return audio_bright

    def process_audio(self,
                     audio: np.ndarray,
                     reference_audio: Optional[np.ndarray] = None,
                     spectral_match_strength: float = 0.7,
                     target_centroid_hz: float = 2800.0) -> Tuple[np.ndarray, Dict]:
        """
        Apply complete post-processing pipeline

        Args:
            audio: Input audio
            reference_audio: Optional reference for spectral matching
            spectral_match_strength: Spectral matching strength (0-1)
            target_centroid_hz: Target brightness in Hz

        Returns:
            Tuple of (processed_audio, metrics_dict)
        """
        logger.info("Processing audio...")

        # Calculate original metrics
        original_lufs = self.calculate_lufs(audio)
        original_peak = np.max(np.abs(audio))
        original_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        )

        # Step 1: Loudness normalization
        audio = self.normalize_loudness(audio)

        # Step 2: Spectral matching (if reference provided)
        if reference_audio is not None:
            audio = self.apply_spectral_matching(
                audio,
                reference_audio,
                strength=spectral_match_strength
            )

        # Step 3: Targeted brightness enhancement
        audio = self.enhance_brightness_targeted(audio, target_centroid_hz)

        # Step 4: True-peak limiting
        audio = self.apply_true_peak_limiter(audio, threshold_db=self.target_peak_db)

        # Calculate final metrics
        final_lufs = self.calculate_lufs(audio)
        final_peak = np.max(np.abs(audio))
        final_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        )

        metrics = {
            'original': {
                'lufs': original_lufs,
                'peak_db': self.linear_to_db(original_peak),
                'centroid_hz': original_centroid
            },
            'final': {
                'lufs': final_lufs,
                'peak_db': self.linear_to_db(final_peak),
                'centroid_hz': final_centroid
            }
        }

        logger.info("✓ Processing complete")

        return audio, metrics

    def process_file(self,
                    input_path: Path,
                    output_path: Path,
                    reference_path: Optional[Path] = None,
                    spectral_match_strength: float = 0.7,
                    target_centroid_hz: float = 2800.0) -> Dict:
        """
        Process a single audio file

        Args:
            input_path: Input file path
            output_path: Output file path
            reference_path: Optional reference file for spectral matching
            spectral_match_strength: Spectral matching strength
            target_centroid_hz: Target brightness

        Returns:
            Metrics dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {input_path.name}")
        logger.info('='*60)

        # Load audio
        audio, sr = librosa.load(str(input_path), sr=self.sr, mono=True)

        # Load reference if provided
        reference_audio = None
        if reference_path and reference_path.exists():
            reference_audio, _ = librosa.load(str(reference_path), sr=self.sr, mono=True)
            logger.info(f"Using reference: {reference_path.name}")

        # Process
        audio_processed, metrics = self.process_audio(
            audio,
            reference_audio,
            spectral_match_strength,
            target_centroid_hz
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        sf.write(str(output_path), audio_processed, self.sr, subtype='PCM_16')
        logger.info(f"✓ Saved: {output_path.name}\n")

        return metrics

    def process_directory(self,
                         input_dir: Path,
                         output_dir: Path,
                         reference_dir: Optional[Path] = None,
                         pattern: str = "*.wav",
                         spectral_match_strength: float = 0.7,
                         target_centroid_hz: float = 2800.0) -> None:
        """
        Process all files in a directory

        Args:
            input_dir: Input directory
            output_dir: Output directory
            reference_dir: Optional directory with reference files
            pattern: File pattern to match
            spectral_match_strength: Spectral matching strength
            target_centroid_hz: Target brightness
        """
        input_files = list(input_dir.glob(pattern))

        if not input_files:
            logger.warning(f"No files matching '{pattern}' found in {input_dir}")
            return

        logger.info(f"\nProcessing {len(input_files)} files from {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = []

        for input_file in input_files:
            output_file = output_dir / input_file.name

            # Look for matching reference file
            reference_file = None
            if reference_dir:
                reference_file = reference_dir / input_file.name
                if not reference_file.exists():
                    reference_file = None

            try:
                metrics = self.process_file(
                    input_file,
                    output_file,
                    reference_file,
                    spectral_match_strength,
                    target_centroid_hz
                )
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {e}")

        # Print summary
        if all_metrics:
            logger.info("\n" + "="*60)
            logger.info("PROCESSING SUMMARY")
            logger.info("="*60)

            orig_lufs = np.mean([m['original']['lufs'] for m in all_metrics])
            final_lufs = np.mean([m['final']['lufs'] for m in all_metrics])

            orig_peak = np.mean([m['original']['peak_db'] for m in all_metrics])
            final_peak = np.mean([m['final']['peak_db'] for m in all_metrics])

            orig_centroid = np.mean([m['original']['centroid_hz'] for m in all_metrics])
            final_centroid = np.mean([m['final']['centroid_hz'] for m in all_metrics])

            logger.info(f"Files processed: {len(all_metrics)}")
            logger.info(f"\nLoudness:")
            logger.info(f"  Before: {orig_lufs:.1f} LUFS")
            logger.info(f"  After:  {final_lufs:.1f} LUFS (target: {self.target_lufs:.1f})")
            logger.info(f"\nPeak level:")
            logger.info(f"  Before: {orig_peak:.1f} dBFS")
            logger.info(f"  After:  {final_peak:.1f} dBFS (target: {self.target_peak_db:.1f})")
            logger.info(f"\nBrightness:")
            logger.info(f"  Before: {orig_centroid:.0f} Hz")
            logger.info(f"  After:  {final_centroid:.0f} Hz (target: {target_centroid_hz:.0f})")
            logger.info("="*60 + "\n")


def main():
    """Main execution function"""

    # Initialize processor
    processor = ProductionPostProcessor(
        target_lufs=-16.0,      # Modern streaming/broadcast standard
        target_peak_db=-1.0,    # True peak headroom for codec safety
        sr=22050
    )

    base_dir = Path(__file__).parent

    # Process production clone outputs
    input_dir = base_dir / "production_clone_output"
    output_dir = base_dir / "production_clone_output_processed"

    # Use remastered training data as reference
    reference_dir = base_dir / "training_data" / "segments_remastered"

    if not reference_dir.exists():
        logger.warning(f"Reference directory not found: {reference_dir}")
        logger.info("Using default brightness enhancement without spectral matching")
        reference_dir = None

    if input_dir.exists():
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION AUDIO POST-PROCESSING")
        logger.info("="*60)

        # Calculate average brightness from training data
        target_centroid = 2800.0  # Default

        if reference_dir and reference_dir.exists():
            logger.info("Analyzing reference training data...")
            ref_files = list(reference_dir.glob("*.wav"))[:10]  # Sample 10 files
            centroids = []

            for ref_file in ref_files:
                try:
                    audio, sr = librosa.load(str(ref_file), sr=22050, mono=True)
                    centroid = np.mean(
                        librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                    )
                    centroids.append(centroid)
                except:
                    pass

            if centroids:
                target_centroid = np.mean(centroids)
                logger.info(f"Target brightness from training data: {target_centroid:.0f} Hz")

        processor.process_directory(
            input_dir,
            output_dir,
            reference_dir=reference_dir,
            spectral_match_strength=0.7,  # 70% spectral matching
            target_centroid_hz=target_centroid
        )

        logger.info("\n✓ Post-processing complete!")
        logger.info(f"\nProcessed files: {output_dir}")

    else:
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("\nPlease generate production outputs first using:")
        logger.info("  python production_voice_clone.py")


if __name__ == "__main__":
    main()
