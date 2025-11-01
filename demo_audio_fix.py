"""
Demo: Audio Quality Fix - Before/After Comparison

Creates sample outputs showing the effects of:
1. Training data remastering
2. Production post-processing
3. Side-by-side quality metrics comparison

Usage:
    python demo_audio_fix.py
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from remaster_training_data import TrainingDataRemaster
from production_post_process import ProductionPostProcessor
from audio_quality_checker import AudioQualityChecker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_waveform_comparison(audio_before, audio_after, sr, title, output_path):
    """
    Plot waveform comparison

    Args:
        audio_before: Original audio
        audio_after: Processed audio
        sr: Sample rate
        title: Plot title
        output_path: Output image path
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    time_before = np.arange(len(audio_before)) / sr
    time_after = np.arange(len(audio_after)) / sr

    # Before
    axes[0].plot(time_before, audio_before, linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'{title} - BEFORE', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].axhline(y=0.99, color='r', linestyle='--', linewidth=1, label='Clipping threshold')
    axes[0].axhline(y=-0.99, color='r', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # After
    axes[1].plot(time_after, audio_after, linewidth=0.5, alpha=0.7, color='green')
    axes[1].set_title(f'{title} - AFTER', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].axhline(y=0.95, color='orange', linestyle='--', linewidth=1, label='Headroom')
    axes[1].axhline(y=-0.95, color='orange', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved waveform comparison: {output_path}")


def plot_spectrum_comparison(audio_before, audio_after, sr, title, output_path):
    """
    Plot frequency spectrum comparison

    Args:
        audio_before: Original audio
        audio_after: Processed audio
        sr: Sample rate
        title: Plot title
        output_path: Output image path
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Calculate spectra
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Before
    stft_before = librosa.stft(audio_before, n_fft=2048)
    mag_before = np.abs(stft_before)
    mag_before_db = librosa.amplitude_to_db(mag_before, ref=np.max)
    avg_spectrum_before = np.mean(mag_before_db, axis=1)

    axes[0].plot(freqs, avg_spectrum_before, linewidth=2)
    axes[0].set_title(f'{title} - BEFORE', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[0].set_xlim(0, sr/2)
    axes[0].set_ylim(-80, 0)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=3000, color='r', linestyle='--', alpha=0.5, label='~3000 Hz (before)')
    axes[0].legend()

    # After
    stft_after = librosa.stft(audio_after, n_fft=2048)
    mag_after = np.abs(stft_after)
    mag_after_db = librosa.amplitude_to_db(mag_after, ref=np.max)
    avg_spectrum_after = np.mean(mag_after_db, axis=1)

    axes[1].plot(freqs, avg_spectrum_after, linewidth=2, color='green')
    axes[1].set_title(f'{title} - AFTER', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[1].set_xlim(0, sr/2)
    axes[1].set_ylim(-80, 0)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=2800, color='orange', linestyle='--', alpha=0.5, label='~2800 Hz (target)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved spectrum comparison: {output_path}")


def create_demo_samples():
    """Create demo audio samples showing before/after processing"""

    logger.info("="*80)
    logger.info("AUDIO QUALITY FIX DEMONSTRATION")
    logger.info("="*80 + "\n")

    base_dir = Path(__file__).parent
    demo_dir = base_dir / "demo_samples"
    demo_dir.mkdir(exist_ok=True)

    # Initialize processors
    remaster = TrainingDataRemaster(
        target_rms_db=-21.0,
        target_peak_db=-3.0,
        target_crest_db=8.0,
        sr=22050
    )

    postprocessor = ProductionPostProcessor(
        target_lufs=-16.0,
        target_peak_db=-1.0,
        sr=22050
    )

    checker = AudioQualityChecker()

    # ========================================================================
    # DEMO 1: Training Data Remastering
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: TRAINING DATA REMASTERING")
    logger.info("="*80 + "\n")

    # Find a training segment with clipping
    segments_dir = base_dir / "training_data" / "segments_final_merged"

    if segments_dir.exists():
        segment_files = list(segments_dir.glob("*.wav"))[:3]  # Process first 3

        for i, input_file in enumerate(segment_files, 1):
            logger.info(f"\nProcessing training sample {i}: {input_file.name}")

            # Load original
            audio_orig, sr = librosa.load(str(input_file), sr=22050, mono=True)

            # Save original for comparison
            orig_path = demo_dir / f"training_sample_{i}_BEFORE.wav"
            sf.write(str(orig_path), audio_orig, sr, subtype='PCM_16')

            # Remaster
            audio_remastered, metrics = remaster.remaster_audio(audio_orig)

            # Save remastered
            remastered_path = demo_dir / f"training_sample_{i}_AFTER_remastered.wav"
            sf.write(str(remastered_path), audio_remastered, sr, subtype='PCM_16')

            # Create waveform comparison
            plot_waveform_comparison(
                audio_orig, audio_remastered, sr,
                f"Training Sample {i} Remastering",
                demo_dir / f"training_sample_{i}_waveform.png"
            )

            # Create spectrum comparison
            plot_spectrum_comparison(
                audio_orig, audio_remastered, sr,
                f"Training Sample {i} Spectrum",
                demo_dir / f"training_sample_{i}_spectrum.png"
            )

            # Print metrics
            logger.info(f"\n  Metrics for sample {i}:")
            logger.info(f"    Original:    RMS={metrics['original']['rms_db']:.1f}dB, "
                       f"Peak={metrics['original']['peak_db']:.1f}dB, "
                       f"Crest={metrics['original']['crest_db']:.1f}dB")
            logger.info(f"    Remastered:  RMS={metrics['remastered']['rms_db']:.1f}dB, "
                       f"Peak={metrics['remastered']['peak_db']:.1f}dB, "
                       f"Crest={metrics['remastered']['crest_db']:.1f}dB")

    # ========================================================================
    # DEMO 2: Production Post-Processing
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: PRODUCTION POST-PROCESSING")
    logger.info("="*80 + "\n")

    # Check for production outputs
    production_dir = base_dir / "production_clone_output"

    if production_dir.exists():
        production_files = list(production_dir.glob("*.wav"))[:2]  # Process first 2

        # Get a reference from remastered training data
        reference_path = None
        remastered_dir = base_dir / "training_data" / "segments_remastered"
        if remastered_dir.exists():
            ref_files = list(remastered_dir.glob("*.wav"))
            if ref_files:
                reference_path = ref_files[0]

        for i, input_file in enumerate(production_files, 1):
            logger.info(f"\nProcessing production sample {i}: {input_file.name}")

            # Load original
            audio_orig, sr = librosa.load(str(input_file), sr=22050, mono=True)

            # Save original for comparison
            orig_path = demo_dir / f"production_sample_{i}_BEFORE.wav"
            sf.write(str(orig_path), audio_orig, sr, subtype='PCM_16')

            # Post-process
            reference_audio = None
            if reference_path:
                reference_audio, _ = librosa.load(str(reference_path), sr=22050, mono=True)

            audio_processed, metrics = postprocessor.process_audio(
                audio_orig,
                reference_audio=reference_audio,
                spectral_match_strength=0.7,
                target_centroid_hz=2800.0
            )

            # Save processed
            processed_path = demo_dir / f"production_sample_{i}_AFTER_processed.wav"
            sf.write(str(processed_path), audio_processed, sr, subtype='PCM_16')

            # Create waveform comparison
            plot_waveform_comparison(
                audio_orig, audio_processed, sr,
                f"Production Sample {i} Post-Processing",
                demo_dir / f"production_sample_{i}_waveform.png"
            )

            # Create spectrum comparison
            plot_spectrum_comparison(
                audio_orig, audio_processed, sr,
                f"Production Sample {i} Spectrum",
                demo_dir / f"production_sample_{i}_spectrum.png"
            )

            # Print metrics
            logger.info(f"\n  Metrics for sample {i}:")
            logger.info(f"    Original:  Loudness={metrics['original']['lufs']:.1f}LUFS, "
                       f"Peak={metrics['original']['peak_db']:.1f}dB, "
                       f"Brightness={metrics['original']['centroid_hz']:.0f}Hz")
            logger.info(f"    Processed: Loudness={metrics['final']['lufs']:.1f}LUFS, "
                       f"Peak={metrics['final']['peak_db']:.1f}dB, "
                       f"Brightness={metrics['final']['centroid_hz']:.0f}Hz")

    else:
        logger.warning("Production output directory not found. Skipping demo 2.")
        logger.info("To generate production samples, run: python optimal_voice_clone.py")

    # ========================================================================
    # DEMO 3: Quality Control Report
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: QUALITY CONTROL COMPARISON")
    logger.info("="*80 + "\n")

    # Analyze demo samples
    logger.info("Analyzing BEFORE samples...")
    before_files = list(demo_dir.glob("*_BEFORE.wav"))
    before_metrics = []
    for f in before_files:
        m = checker.analyze_file(f)
        before_metrics.append(m)

    logger.info("\nAnalyzing AFTER samples...")
    after_files = list(demo_dir.glob("*_AFTER_*.wav"))
    after_metrics = []
    for f in after_files:
        m = checker.analyze_file(f)
        after_metrics.append(m)

    # Print comparison table
    if before_metrics and after_metrics:
        logger.info("\n" + "="*80)
        logger.info("BEFORE vs AFTER COMPARISON")
        logger.info("="*80)

        avg_before = {
            'rms': np.mean([m.rms_db for m in before_metrics]),
            'peak': np.mean([m.peak_db for m in before_metrics]),
            'crest': np.mean([m.crest_db for m in before_metrics]),
            'brightness': np.mean([m.centroid_hz for m in before_metrics]),
        }

        avg_after = {
            'rms': np.mean([m.rms_db for m in after_metrics]),
            'peak': np.mean([m.peak_db for m in after_metrics]),
            'crest': np.mean([m.crest_db for m in after_metrics]),
            'brightness': np.mean([m.centroid_hz for m in after_metrics]),
        }

        logger.info(f"\nAverage Metrics:")
        logger.info(f"  RMS Level:   {avg_before['rms']:>7.1f} dBFS  →  {avg_after['rms']:>7.1f} dBFS  "
                   f"({avg_after['rms']-avg_before['rms']:+.1f} dB)")
        logger.info(f"  Peak Level:  {avg_before['peak']:>7.1f} dBFS  →  {avg_after['peak']:>7.1f} dBFS  "
                   f"({avg_after['peak']-avg_before['peak']:+.1f} dB)")
        logger.info(f"  Crest:       {avg_before['crest']:>7.1f} dB    →  {avg_after['crest']:>7.1f} dB    "
                   f"({avg_after['crest']-avg_before['crest']:+.1f} dB)")
        logger.info(f"  Brightness:  {avg_before['brightness']:>7.0f} Hz    →  {avg_after['brightness']:>7.0f} Hz    "
                   f"({avg_after['brightness']-avg_before['brightness']:+.0f} Hz)")

        logger.info("\n" + "="*80)
        logger.info("QUALITY ISSUES")
        logger.info("="*80)

        logger.info(f"\nBEFORE: {sum(m.has_issues for m in before_metrics)}/{len(before_metrics)} files with issues")
        for m in before_metrics:
            if m.has_issues:
                logger.info(f"  {m.filename}: {m.issue_summary}")

        logger.info(f"\nAFTER:  {sum(m.has_issues for m in after_metrics)}/{len(after_metrics)} files with issues")
        for m in after_metrics:
            if m.has_issues:
                logger.info(f"  {m.filename}: {m.issue_summary}")

    # ========================================================================
    # Final Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE!")
    logger.info("="*80)

    logger.info(f"\nDemo samples saved to: {demo_dir}")
    logger.info("\nGenerated files:")

    audio_files = sorted(demo_dir.glob("*.wav"))
    image_files = sorted(demo_dir.glob("*.png"))

    logger.info("\n  Audio samples:")
    for f in audio_files:
        logger.info(f"    - {f.name}")

    logger.info("\n  Visualizations:")
    for f in image_files:
        logger.info(f"    - {f.name}")

    logger.info("\n" + "="*80)
    logger.info("Next Steps:")
    logger.info("="*80)
    logger.info("\n1. Listen to BEFORE vs AFTER samples to hear the improvements")
    logger.info("2. View waveform/spectrum PNG files to see visual differences")
    logger.info("3. Run full pipeline: python run_audio_pipeline.py --all")
    logger.info("4. Update voice cloning scripts to use remastered data")
    logger.info("5. Regenerate all production samples\n")


if __name__ == "__main__":
    try:
        create_demo_samples()
    except Exception as e:
        logger.error(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        logger.info("\nPlease ensure you have:")
        logger.info("1. Training data in training_data/segments_final_merged/")
        logger.info("2. matplotlib installed: pip install matplotlib")
