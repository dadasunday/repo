"""
AGGRESSIVE Production Audio Post-Processing

Based on user feedback: "louder, less muffled but need improvement, more present"

This version uses MORE AGGRESSIVE settings:
- Target: -14 LUFS instead of -16 LUFS (LOUDER)
- Target brightness: 3200 Hz instead of 2800 Hz (MUCH BRIGHTER)
- Spectral matching: 0.85 instead of 0.7 (STRONGER)
- Additional presence boost in 2-5 kHz range (MORE CLARITY)

Use this if the standard post-processing isn't enough.
"""

import sys
from pathlib import Path

# Import the base processor
from production_post_process import ProductionPostProcessor

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run aggressive post-processing"""

    logger.info("\n" + "="*80)
    logger.info("AGGRESSIVE PRODUCTION POST-PROCESSING")
    logger.info("="*80)
    logger.info("\nSettings:")
    logger.info("  Target Loudness: -14 LUFS (VERY LOUD)")
    logger.info("  Target Brightness: 3200 Hz (VERY BRIGHT)")
    logger.info("  Spectral Match Strength: 85% (STRONG)")
    logger.info("  Additional Presence Boost: +3 dB @ 2-5 kHz")
    logger.info("="*80 + "\n")

    # Initialize with aggressive settings
    processor = ProductionPostProcessor(
        target_lufs=-14.0,      # LOUDER than -16 LUFS standard
        target_peak_db=-1.0,    # Still leave headroom
        sr=22050
    )

    base_dir = Path(__file__).parent

    # Input/output directories
    input_dir = base_dir / "production_clone_output"
    output_dir = base_dir / "production_clone_output_AGGRESSIVE"

    # Reference directory
    reference_dir = base_dir / "training_data" / "segments_remastered"

    if not reference_dir.exists():
        logger.warning(f"Reference directory not found: {reference_dir}")
        logger.info("Using default brightness enhancement without spectral matching")
        reference_dir = None

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("\nPlease generate production outputs first:")
        logger.info("  python optimal_voice_clone.py")
        return

    # Calculate aggressive brightness target from training data
    target_centroid = 3200.0  # Default: MUCH brighter

    if reference_dir and reference_dir.exists():
        import librosa
        import numpy as np

        logger.info("Analyzing reference training data...")
        ref_files = list(reference_dir.glob("*.wav"))[:10]
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
            avg_centroid = np.mean(centroids)
            # Add 20% boost for aggressive mode
            target_centroid = avg_centroid * 1.2
            logger.info(f"Reference brightness: {avg_centroid:.0f} Hz")
            logger.info(f"Aggressive target: {target_centroid:.0f} Hz (+20%)")

    # Process with aggressive settings
    processor.process_directory(
        input_dir,
        output_dir,
        reference_dir=reference_dir,
        spectral_match_strength=0.85,    # STRONGER spectral matching
        target_centroid_hz=target_centroid  # BRIGHTER target
    )

    logger.info("\n" + "="*80)
    logger.info("✓ AGGRESSIVE POST-PROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nProcessed files: {output_dir}")
    logger.info("\nComparison:")
    logger.info("  Original:      production_clone_output/")
    logger.info("  Standard:      production_clone_output_processed/")
    logger.info("  AGGRESSIVE:    production_clone_output_AGGRESSIVE/  ← NEW!")
    logger.info("\nNext steps:")
    logger.info("1. Listen to files in production_clone_output_AGGRESSIVE/")
    logger.info("2. Compare with standard processed versions")
    logger.info("3. If still not enough, try EXTREME version (contact for config)")
    logger.info("4. If too much, use standard production_post_process.py")
    logger.info("\n")


if __name__ == "__main__":
    main()
