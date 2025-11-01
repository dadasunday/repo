"""
Automated Audio Quality Control Script

Analyzes audio files for technical quality metrics and flags outliers.
Checks RMS, peak, crest factor, spectral centroid, zero-crossing rate,
and noise floor to ensure production-ready quality.

Usage:
    python audio_quality_checker.py <directory>
    python audio_quality_checker.py --compare-dirs <dir1> <dir2>
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Quality thresholds for audio validation"""
    # Loudness
    min_rms_db: float = -30.0
    max_rms_db: float = -6.0
    target_rms_db: float = -21.0
    rms_tolerance_db: float = 6.0

    # Peak levels
    max_peak_db: float = -0.5       # Should leave headroom
    clip_threshold: float = -0.1    # Flag if peaks are this close to 0 dBFS

    # Dynamic range
    min_crest_db: float = 4.0       # Too compressed
    max_crest_db: float = 15.0      # Too dynamic/raw
    target_crest_db: float = 8.0
    crest_tolerance_db: float = 3.0

    # Spectral characteristics
    min_centroid_hz: float = 800.0     # Too dark/muffled
    max_centroid_hz: float = 4500.0    # Too bright/harsh
    target_centroid_hz: float = 2500.0
    centroid_tolerance_hz: float = 800.0

    # Noise and artifacts
    max_zcr_rate: float = 0.2          # High = noisy/harsh
    min_spectral_flatness: float = 0.01  # Pure tone (synthesis artifact?)
    max_spectral_flatness: float = 0.5   # Too noisy


@dataclass
class AudioMetrics:
    """Container for audio quality metrics"""
    filename: str
    duration: float
    sample_rate: int

    # Loudness metrics
    rms_db: float
    peak_db: float
    crest_db: float

    # Spectral metrics
    centroid_hz: float
    bandwidth_hz: float
    rolloff_hz: float
    flatness: float

    # Temporal metrics
    zcr_rate: float

    # Noise metrics
    noise_floor_db: float

    # Quality flags
    has_clipping: bool
    is_too_quiet: bool
    is_too_loud: bool
    is_too_compressed: bool
    is_too_dynamic: bool
    is_too_dark: bool
    is_too_bright: bool
    is_too_noisy: bool
    has_dc_offset: bool

    @property
    def has_issues(self) -> bool:
        """Check if audio has any quality issues"""
        return any([
            self.has_clipping,
            self.is_too_quiet,
            self.is_too_loud,
            self.is_too_compressed,
            self.is_too_dynamic,
            self.is_too_dark,
            self.is_too_bright,
            self.is_too_noisy,
            self.has_dc_offset
        ])

    @property
    def issue_summary(self) -> str:
        """Get human-readable summary of issues"""
        issues = []
        if self.has_clipping:
            issues.append("CLIPPING")
        if self.is_too_quiet:
            issues.append("TOO QUIET")
        if self.is_too_loud:
            issues.append("TOO LOUD")
        if self.is_too_compressed:
            issues.append("OVER-COMPRESSED")
        if self.is_too_dynamic:
            issues.append("TOO DYNAMIC")
        if self.is_too_dark:
            issues.append("TOO DARK")
        if self.is_too_bright:
            issues.append("TOO BRIGHT")
        if self.is_too_noisy:
            issues.append("NOISY")
        if self.has_dc_offset:
            issues.append("DC OFFSET")

        return ", ".join(issues) if issues else "PASS"


class AudioQualityChecker:
    """Analyze and validate audio quality"""

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Args:
            thresholds: Quality thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or QualityThresholds()

    def db_to_linear(self, db: float) -> float:
        """Convert dB to linear amplitude"""
        return 10 ** (db / 20.0)

    def linear_to_db(self, linear: float) -> float:
        """Convert linear amplitude to dB"""
        return 20 * np.log10(max(linear, 1e-10))

    def calculate_noise_floor(self, audio: np.ndarray, percentile: float = 10) -> float:
        """
        Estimate noise floor from quietest portions

        Args:
            audio: Input audio
            percentile: Percentile for noise floor estimation

        Returns:
            Noise floor in dBFS
        """
        # Calculate RMS in small windows
        frame_length = 2048
        hop_length = 512

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        rms_values = np.sqrt(np.mean(frames ** 2, axis=0))

        # Get noise floor from quietest frames
        noise_floor = np.percentile(rms_values, percentile)
        noise_floor_db = self.linear_to_db(noise_floor)

        return noise_floor_db

    def detect_dc_offset(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect DC offset

        Args:
            audio: Input audio
            threshold: DC offset threshold (0-1 range)

        Returns:
            True if significant DC offset detected
        """
        mean = np.abs(np.mean(audio))
        return mean > threshold

    def analyze_audio(self, audio: np.ndarray, sr: int, filename: str = "") -> AudioMetrics:
        """
        Perform complete quality analysis

        Args:
            audio: Input audio
            sr: Sample rate
            filename: Filename for reporting

        Returns:
            AudioMetrics object with all measurements
        """
        # Basic measurements
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        rms_db = self.linear_to_db(rms)
        peak_db = self.linear_to_db(peak)
        crest_db = peak_db - rms_db

        # Spectral features
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio)[0])

        # Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

        # Noise floor
        noise_floor_db = self.calculate_noise_floor(audio)

        # Quality checks
        t = self.thresholds

        has_clipping = peak_db >= t.clip_threshold
        is_too_quiet = rms_db < t.min_rms_db
        is_too_loud = rms_db > t.max_rms_db
        is_too_compressed = crest_db < t.min_crest_db
        is_too_dynamic = crest_db > t.max_crest_db
        is_too_dark = centroid < t.min_centroid_hz
        is_too_bright = centroid > t.max_centroid_hz
        is_too_noisy = (flatness > t.max_spectral_flatness or
                       zcr > t.max_zcr_rate or
                       noise_floor_db > -40.0)
        has_dc_offset = self.detect_dc_offset(audio)

        return AudioMetrics(
            filename=filename,
            duration=duration,
            sample_rate=sr,
            rms_db=rms_db,
            peak_db=peak_db,
            crest_db=crest_db,
            centroid_hz=centroid,
            bandwidth_hz=bandwidth,
            rolloff_hz=rolloff,
            flatness=flatness,
            zcr_rate=zcr,
            noise_floor_db=noise_floor_db,
            has_clipping=has_clipping,
            is_too_quiet=is_too_quiet,
            is_too_loud=is_too_loud,
            is_too_compressed=is_too_compressed,
            is_too_dynamic=is_too_dynamic,
            is_too_dark=is_too_dark,
            is_too_bright=is_too_bright,
            is_too_noisy=is_too_noisy,
            has_dc_offset=has_dc_offset
        )

    def analyze_file(self, filepath: Path) -> AudioMetrics:
        """
        Analyze a single audio file

        Args:
            filepath: Path to audio file

        Returns:
            AudioMetrics object
        """
        audio, sr = librosa.load(str(filepath), sr=None, mono=True)
        return self.analyze_audio(audio, sr, filename=filepath.name)

    def analyze_directory(self, directory: Path, pattern: str = "*.wav") -> List[AudioMetrics]:
        """
        Analyze all audio files in a directory

        Args:
            directory: Directory path
            pattern: File pattern to match

        Returns:
            List of AudioMetrics
        """
        files = list(directory.glob(pattern))

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return []

        logger.info(f"\nAnalyzing {len(files)} files in {directory}...\n")

        results = []
        for filepath in files:
            try:
                metrics = self.analyze_file(filepath)
                results.append(metrics)

                # Log issues immediately
                if metrics.has_issues:
                    logger.warning(f"⚠️  {metrics.filename}: {metrics.issue_summary}")
                else:
                    logger.info(f"✓  {metrics.filename}: PASS")

            except Exception as e:
                logger.error(f"Error analyzing {filepath.name}: {e}")

        return results

    def print_summary_report(self, metrics_list: List[AudioMetrics]) -> None:
        """
        Print detailed summary report

        Args:
            metrics_list: List of AudioMetrics to summarize
        """
        if not metrics_list:
            return

        logger.info("\n" + "="*80)
        logger.info("QUALITY ANALYSIS SUMMARY")
        logger.info("="*80)

        # Overall statistics
        total = len(metrics_list)
        passed = sum(1 for m in metrics_list if not m.has_issues)
        failed = total - passed

        logger.info(f"\nTotal files analyzed: {total}")
        logger.info(f"Passed QC: {passed} ({100*passed/total:.1f}%)")
        logger.info(f"Failed QC: {failed} ({100*failed/total:.1f}%)")

        # Average metrics
        avg_rms = np.mean([m.rms_db for m in metrics_list])
        avg_peak = np.mean([m.peak_db for m in metrics_list])
        avg_crest = np.mean([m.crest_db for m in metrics_list])
        avg_centroid = np.mean([m.centroid_hz for m in metrics_list])
        avg_zcr = np.mean([m.zcr_rate for m in metrics_list])
        avg_noise = np.mean([m.noise_floor_db for m in metrics_list])

        logger.info(f"\nAverage Metrics:")
        logger.info(f"  RMS:           {avg_rms:>7.1f} dBFS  (target: {self.thresholds.target_rms_db:.1f} dBFS)")
        logger.info(f"  Peak:          {avg_peak:>7.1f} dBFS  (max: {self.thresholds.max_peak_db:.1f} dBFS)")
        logger.info(f"  Crest Factor:  {avg_crest:>7.1f} dB    (target: {self.thresholds.target_crest_db:.1f} dB)")
        logger.info(f"  Brightness:    {avg_centroid:>7.0f} Hz    (target: {self.thresholds.target_centroid_hz:.0f} Hz)")
        logger.info(f"  ZCR:           {avg_zcr:>7.3f}      (max: {self.thresholds.max_zcr_rate:.3f})")
        logger.info(f"  Noise Floor:   {avg_noise:>7.1f} dBFS")

        # Issue breakdown
        if failed > 0:
            logger.info(f"\nIssue Breakdown:")
            issue_counts = {
                'Clipping': sum(m.has_clipping for m in metrics_list),
                'Too Quiet': sum(m.is_too_quiet for m in metrics_list),
                'Too Loud': sum(m.is_too_loud for m in metrics_list),
                'Over-compressed': sum(m.is_too_compressed for m in metrics_list),
                'Too Dynamic': sum(m.is_too_dynamic for m in metrics_list),
                'Too Dark': sum(m.is_too_dark for m in metrics_list),
                'Too Bright': sum(m.is_too_bright for m in metrics_list),
                'Noisy': sum(m.is_too_noisy for m in metrics_list),
                'DC Offset': sum(m.has_dc_offset for m in metrics_list)
            }

            for issue, count in issue_counts.items():
                if count > 0:
                    logger.info(f"  {issue:.<20} {count:>3} ({100*count/total:>5.1f}%)")

            # List problematic files
            logger.info(f"\nFiles with Issues:")
            for m in metrics_list:
                if m.has_issues:
                    logger.info(f"  {m.filename}")
                    logger.info(f"    Issues: {m.issue_summary}")
                    logger.info(f"    RMS={m.rms_db:.1f}dB, Peak={m.peak_db:.1f}dB, "
                              f"Crest={m.crest_db:.1f}dB, Brightness={m.centroid_hz:.0f}Hz")

        logger.info("="*80 + "\n")

    def compare_directories(self,
                           dir1: Path,
                           dir2: Path,
                           dir1_label: str = "Directory 1",
                           dir2_label: str = "Directory 2") -> None:
        """
        Compare metrics between two directories

        Args:
            dir1: First directory
            dir2: Second directory
            dir1_label: Label for first directory
            dir2_label: Label for second directory
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARING: {dir1_label} vs {dir2_label}")
        logger.info(f"{'='*80}\n")

        metrics1 = self.analyze_directory(dir1)
        metrics2 = self.analyze_directory(dir2)

        if not metrics1 or not metrics2:
            logger.error("Cannot compare: one or both directories have no valid files")
            return

        # Calculate averages
        avg1 = {
            'rms_db': np.mean([m.rms_db for m in metrics1]),
            'peak_db': np.mean([m.peak_db for m in metrics1]),
            'crest_db': np.mean([m.crest_db for m in metrics1]),
            'centroid_hz': np.mean([m.centroid_hz for m in metrics1]),
            'zcr': np.mean([m.zcr_rate for m in metrics1]),
        }

        avg2 = {
            'rms_db': np.mean([m.rms_db for m in metrics2]),
            'peak_db': np.mean([m.peak_db for m in metrics2]),
            'crest_db': np.mean([m.crest_db for m in metrics2]),
            'centroid_hz': np.mean([m.centroid_hz for m in metrics2]),
            'zcr': np.mean([m.zcr_rate for m in metrics2]),
        }

        # Print comparison
        logger.info("\n" + "="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)
        logger.info(f"\n{'Metric':<20} {dir1_label:<20} {dir2_label:<20} {'Difference':<15}")
        logger.info("-"*80)

        metrics_to_compare = [
            ('RMS (dBFS)', 'rms_db', 'dB'),
            ('Peak (dBFS)', 'peak_db', 'dB'),
            ('Crest Factor', 'crest_db', 'dB'),
            ('Brightness', 'centroid_hz', 'Hz'),
            ('Zero-Cross Rate', 'zcr', ''),
        ]

        for name, key, unit in metrics_to_compare:
            val1 = avg1[key]
            val2 = avg2[key]
            diff = val2 - val1

            # Format values
            if 'hz' in key.lower():
                val1_str = f"{val1:.0f} {unit}"
                val2_str = f"{val2:.0f} {unit}"
                diff_str = f"{diff:+.0f} {unit}"
            elif key == 'zcr':
                val1_str = f"{val1:.4f}"
                val2_str = f"{val2:.4f}"
                diff_str = f"{diff:+.4f}"
            else:
                val1_str = f"{val1:.1f} {unit}"
                val2_str = f"{val2:.1f} {unit}"
                diff_str = f"{diff:+.1f} {unit}"

            logger.info(f"{name:<20} {val1_str:<20} {val2_str:<20} {diff_str:<15}")

        logger.info("="*80 + "\n")

    def export_metrics(self, metrics_list: List[AudioMetrics], output_path: Path) -> None:
        """
        Export metrics to JSON file

        Args:
            metrics_list: List of AudioMetrics
            output_path: Output JSON file path
        """
        data = []
        for m in metrics_list:
            data.append({
                'filename': m.filename,
                'duration': m.duration,
                'sample_rate': m.sample_rate,
                'rms_db': m.rms_db,
                'peak_db': m.peak_db,
                'crest_db': m.crest_db,
                'centroid_hz': m.centroid_hz,
                'bandwidth_hz': m.bandwidth_hz,
                'rolloff_hz': m.rolloff_hz,
                'flatness': m.flatness,
                'zcr_rate': m.zcr_rate,
                'noise_floor_db': m.noise_floor_db,
                'has_issues': m.has_issues,
                'issues': m.issue_summary
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to: {output_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Audio Quality Control Checker')
    parser.add_argument('directory', nargs='?', type=str, help='Directory to analyze')
    parser.add_argument('--compare', nargs=2, metavar=('DIR1', 'DIR2'),
                       help='Compare two directories')
    parser.add_argument('--export', type=str, help='Export metrics to JSON file')
    parser.add_argument('--pattern', type=str, default='*.wav', help='File pattern (default: *.wav)')

    args = parser.parse_args()

    checker = AudioQualityChecker()

    if args.compare:
        # Compare two directories
        dir1 = Path(args.compare[0])
        dir2 = Path(args.compare[1])

        checker.compare_directories(
            dir1, dir2,
            dir1_label=dir1.name,
            dir2_label=dir2.name
        )

    elif args.directory:
        # Analyze single directory
        directory = Path(args.directory)

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return

        metrics = checker.analyze_directory(directory, pattern=args.pattern)
        checker.print_summary_report(metrics)

        if args.export:
            checker.export_metrics(metrics, Path(args.export))

    else:
        # Default: analyze common directories
        base_dir = Path(__file__).parent

        dirs_to_check = [
            ('Training Data (Original)', base_dir / 'training_data' / 'segments_final_merged'),
            ('Training Data (Remastered)', base_dir / 'training_data' / 'segments_remastered'),
            ('Production Output (Original)', base_dir / 'production_clone_output'),
            ('Production Output (Processed)', base_dir / 'production_clone_output_processed'),
        ]

        for label, directory in dirs_to_check:
            if directory.exists():
                logger.info(f"\n{'='*80}")
                logger.info(f"Checking: {label}")
                logger.info(f"{'='*80}")

                metrics = checker.analyze_directory(directory)
                checker.print_summary_report(metrics)


if __name__ == "__main__":
    main()
