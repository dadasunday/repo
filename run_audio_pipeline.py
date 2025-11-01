"""
Complete Audio Quality Pipeline Runner

Automates the entire audio fix workflow:
1. Remaster training data
2. Improve segmentation
3. Post-process production outputs
4. Run quality control checks

Usage:
    python run_audio_pipeline.py --all              # Run complete pipeline
    python run_audio_pipeline.py --remaster         # Step 1 only
    python run_audio_pipeline.py --segment          # Step 2 only
    python run_audio_pipeline.py --postprocess      # Step 3 only
    python run_audio_pipeline.py --qc               # Step 4 only
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPipeline:
    """Complete audio quality pipeline orchestrator"""

    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = base_dir or Path(__file__).parent
        self.training_dir = self.base_dir / "training_data"

    def print_banner(self, title: str) -> None:
        """Print section banner"""
        logger.info("\n" + "="*80)
        logger.info(title.center(80))
        logger.info("="*80 + "\n")

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        Check if required directories and files exist

        Returns:
            Tuple of (all_ok, issues)
        """
        issues = []

        # Check training data directory
        if not self.training_dir.exists():
            issues.append(f"Training data directory not found: {self.training_dir}")

        # Check for training segments
        segments_dir = self.training_dir / "segments_final_merged"
        if not segments_dir.exists():
            issues.append(f"Training segments not found: {segments_dir}")
        elif len(list(segments_dir.glob("*.wav"))) == 0:
            issues.append(f"No WAV files in: {segments_dir}")

        # Check for production outputs
        production_dir = self.base_dir / "production_clone_output"
        if not production_dir.exists():
            logger.warning(f"Production output directory not found: {production_dir}")
            logger.warning("Post-processing step will be skipped")

        return (len(issues) == 0, issues)

    def step1_remaster_training(self) -> bool:
        """
        Step 1: Remaster training data

        Returns:
            True if successful
        """
        self.print_banner("STEP 1: REMASTERING TRAINING DATA")

        try:
            from remaster_training_data import TrainingDataRemaster, main as remaster_main

            logger.info("Importing remaster module...")
            remaster_main()

            # Verify output
            output_dir = self.training_dir / "segments_remastered"
            if output_dir.exists():
                count = len(list(output_dir.glob("*.wav")))
                logger.info(f"✓ Created {count} remastered segments")
                return True
            else:
                logger.error("✗ Remastered directory not created")
                return False

        except ImportError as e:
            logger.error(f"✗ Failed to import remaster module: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Remastering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step2_improve_segmentation(self) -> bool:
        """
        Step 2: Improve segmentation

        Returns:
            True if successful
        """
        self.print_banner("STEP 2: IMPROVING SEGMENTATION")

        try:
            from improved_segmentation import ImprovedSegmentation, SegmentConfig, main as segment_main

            logger.info("Running improved segmentation...")
            segment_main()

            # Verify output
            output_dir = self.training_dir / "segments_improved"
            if output_dir.exists():
                count = len(list(output_dir.glob("*.wav")))
                logger.info(f"✓ Created {count} improved segments")

                # Optional: merge with remastered
                remastered_dir = self.training_dir / "segments_remastered"
                if remastered_dir.exists():
                    logger.info("\nMerging improved segments with remastered data...")
                    logger.info("You can manually copy files if desired:")
                    logger.info(f"  From: {output_dir}")
                    logger.info(f"  To:   {remastered_dir}")

                return True
            else:
                logger.error("✗ Improved segments directory not created")
                return False

        except ImportError as e:
            logger.error(f"✗ Failed to import segmentation module: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step3_postprocess_production(self) -> bool:
        """
        Step 3: Post-process production outputs

        Returns:
            True if successful
        """
        self.print_banner("STEP 3: POST-PROCESSING PRODUCTION OUTPUTS")

        production_dir = self.base_dir / "production_clone_output"

        if not production_dir.exists():
            logger.warning(f"✗ Production directory not found: {production_dir}")
            logger.warning("Skipping post-processing step")
            logger.info("\nTo generate production outputs, run:")
            logger.info("  python optimal_voice_clone.py")
            logger.info("  or")
            logger.info("  python production_voice_clone.py")
            return False

        try:
            from production_post_process import ProductionPostProcessor, main as postprocess_main

            logger.info("Running production post-processing...")
            postprocess_main()

            # Verify output
            output_dir = self.base_dir / "production_clone_output_processed"
            if output_dir.exists():
                count = len(list(output_dir.glob("*.wav")))
                logger.info(f"✓ Processed {count} production files")
                return True
            else:
                logger.error("✗ Processed output directory not created")
                return False

        except ImportError as e:
            logger.error(f"✗ Failed to import post-processing module: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step4_quality_control(self) -> bool:
        """
        Step 4: Run quality control checks

        Returns:
            True if successful
        """
        self.print_banner("STEP 4: QUALITY CONTROL CHECKS")

        try:
            from audio_quality_checker import AudioQualityChecker

            checker = AudioQualityChecker()

            # Check remastered training data
            remastered_dir = self.training_dir / "segments_remastered"
            if remastered_dir.exists():
                logger.info("Analyzing remastered training data...")
                metrics = checker.analyze_directory(remastered_dir)
                checker.print_summary_report(metrics)

                # Export metrics
                export_path = remastered_dir.parent / "remastered_metrics.json"
                checker.export_metrics(metrics, export_path)
                logger.info(f"Metrics exported to: {export_path}")

            # Check processed production outputs
            processed_dir = self.base_dir / "production_clone_output_processed"
            if processed_dir.exists():
                logger.info("\nAnalyzing processed production outputs...")
                metrics = checker.analyze_directory(processed_dir)
                checker.print_summary_report(metrics)

                # Export metrics
                export_path = processed_dir.parent / "processed_metrics.json"
                checker.export_metrics(metrics, export_path)
                logger.info(f"Metrics exported to: {export_path}")

                # Compare training vs production
                if remastered_dir.exists():
                    logger.info("\nComparing training vs production...")
                    checker.compare_directories(
                        remastered_dir,
                        processed_dir,
                        dir1_label="Training (Remastered)",
                        dir2_label="Production (Processed)"
                    )

            return True

        except ImportError as e:
            logger.error(f"✗ Failed to import QC module: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Quality control failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_complete_pipeline(self) -> None:
        """Run the complete audio pipeline"""
        self.print_banner("COMPLETE AUDIO QUALITY PIPELINE")

        # Check prerequisites
        logger.info("Checking prerequisites...")
        ok, issues = self.check_prerequisites()

        if not ok:
            logger.error("✗ Prerequisites check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            logger.info("\nPlease ensure:")
            logger.info("1. Training data exists in training_data/")
            logger.info("2. Segments exist in training_data/segments_final_merged/")
            sys.exit(1)

        logger.info("✓ Prerequisites check passed\n")

        # Track results
        results = {}

        # Step 1: Remaster training data
        results['remaster'] = self.step1_remaster_training()

        # Step 2: Improve segmentation
        results['segment'] = self.step2_improve_segmentation()

        # Step 3: Post-process production (if available)
        results['postprocess'] = self.step3_postprocess_production()

        # Step 4: Quality control
        results['qc'] = self.step4_quality_control()

        # Final summary
        self.print_banner("PIPELINE EXECUTION SUMMARY")

        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        logger.info(f"Completed steps: {success_count}/{total_count}")
        logger.info("")

        for step, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"  {step.capitalize():.<30} {status}")

        logger.info("")

        if success_count == total_count:
            logger.info("✓✓✓ Pipeline completed successfully! ✓✓✓")
            logger.info("\nNext steps:")
            logger.info("1. Review QC reports and metrics")
            logger.info("2. Update voice cloning scripts to use segments_remastered/")
            logger.info("3. Regenerate voice samples with cleaned data")
            logger.info("4. Compare new outputs with original using QC script")
        else:
            logger.error("✗✗✗ Pipeline completed with errors ✗✗✗")
            logger.info("\nPlease review error messages above and fix issues")

        logger.info("\n" + "="*80 + "\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Complete Audio Quality Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_audio_pipeline.py --all              # Run complete pipeline
  python run_audio_pipeline.py --remaster         # Step 1 only
  python run_audio_pipeline.py --segment          # Step 2 only
  python run_audio_pipeline.py --postprocess      # Step 3 only
  python run_audio_pipeline.py --qc               # Step 4 only

Steps:
  1. Remaster:     Remove clipping, normalize training data
  2. Segment:      Create clean, consistent segments
  3. Post-process: Master production outputs with LUFS/spectral matching
  4. QC:           Validate quality and compare metrics

For detailed documentation, see AUDIO_FIX_WORKFLOW.md
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (all steps)')
    parser.add_argument('--remaster', action='store_true',
                       help='Step 1: Remaster training data')
    parser.add_argument('--segment', action='store_true',
                       help='Step 2: Improve segmentation')
    parser.add_argument('--postprocess', action='store_true',
                       help='Step 3: Post-process production outputs')
    parser.add_argument('--qc', action='store_true',
                       help='Step 4: Run quality control checks')
    parser.add_argument('--dir', type=str, default=None,
                       help='Base directory (default: script directory)')

    args = parser.parse_args()

    # Initialize pipeline
    base_dir = Path(args.dir) if args.dir else None
    pipeline = AudioPipeline(base_dir)

    # Determine what to run
    if args.all:
        # Run complete pipeline
        pipeline.run_complete_pipeline()

    elif args.remaster or args.segment or args.postprocess or args.qc:
        # Run specific steps
        if args.remaster:
            success = pipeline.step1_remaster_training()
            if not success:
                sys.exit(1)

        if args.segment:
            success = pipeline.step2_improve_segmentation()
            if not success:
                sys.exit(1)

        if args.postprocess:
            success = pipeline.step3_postprocess_production()
            if not success:
                sys.exit(1)

        if args.qc:
            success = pipeline.step4_quality_control()
            if not success:
                sys.exit(1)

        logger.info("\n✓ Selected steps completed successfully!\n")

    else:
        # No arguments - show help and offer to run complete pipeline
        parser.print_help()
        print("\n" + "="*80)
        print("No arguments provided. Would you like to run the complete pipeline?")
        print("="*80)
        response = input("\nRun complete pipeline? (y/n): ").strip().lower()

        if response in ['y', 'yes']:
            pipeline.run_complete_pipeline()
        else:
            print("\nExiting. Run with --help for usage information.")
            sys.exit(0)


if __name__ == "__main__":
    main()
