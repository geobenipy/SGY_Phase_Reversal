"""
run_phase_rotation_detector.py - Main Execution Script
=======================================================

Detects phase-rotated reflectors in seismic data for hydrocarbon
indicator analysis and interpretation support.

Usage:
    python run_phase_rotation_detector.py

All settings configured in config.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Import configuration
import config

# Import processing modules
from segy_processor import PhaseRotationDetector
from segy_qc import PhaseRotationQC, StatisticsExporter


def setup_logging():
    """Configure logging system."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    handlers = []
    
    if config.LOG_TO_FILE:
        log_file = config.OUTPUT_DIR / config.LOG_FILENAME
        handlers.append(logging.FileHandler(log_file))
    
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def print_banner(logger):
    """Print startup banner."""
    banner = f"""
{'='*70}
    Phase Rotation Detection for Seismic Interpretation
{'='*70}

Configuration:
  Input Directory:     {config.INPUT_DIR}
  Output Directory:    {config.OUTPUT_DIR}
  
Reference Horizon:
  Type:                {config.REFERENCE_TYPE}
  Method:              {'Seafloor Detection' if config.REFERENCE_TYPE == 'seafloor' else 'Manual/Auto'}
  
Analysis Settings:
  Phase Method:        {config.PHASE_METHOD}
  Window Size:         {config.ANALYSIS_WINDOW} samples
  Smoothing:           {config.SMOOTH_PHASES}
  Threshold:           {config.PHASE_ROTATION_THRESHOLD}°

Output Options:
  Attribute Type:      {config.OUTPUT_ATTRIBUTE}
  Normalize:           {config.NORMALIZE_OUTPUT}
  Apply Gain:          {config.APPLY_GAIN} (factor: {config.GAIN_FACTOR if config.APPLY_GAIN else 'N/A'})
  Create QC Plots:     {config.CREATE_QC_PLOTS}

{'='*70}
    """
    print(banner)
    logger.info("Starting phase rotation detection")


def validate_configuration(logger):
    """Validate configuration and directories."""
    errors = []
    
    # Check input directory
    if not config.INPUT_DIR.exists():
        errors.append(f"Input directory does not exist: {config.INPUT_DIR}")
        logger.error("Please create the input directory or update INPUT_DIR in config.py")
        return False
    
    # Check for SEG-Y files
    segy_files = []
    for ext in config.FILE_EXTENSIONS:
        if config.FILENAME_PATTERN:
            segy_files.extend(
                list(config.INPUT_DIR.glob(f"{config.FILENAME_PATTERN}{ext}"))
            )
        else:
            segy_files.extend(list(config.INPUT_DIR.glob(f"*{ext}")))
    
    if not segy_files:
        errors.append(f"No SEG-Y files found in {config.INPUT_DIR}")
        logger.error(f"Please add .sgy or .segy files to {config.INPUT_DIR}")
        return False
    
    # Validate settings
    if config.PHASE_ROTATION_THRESHOLD < 0 or config.PHASE_ROTATION_THRESHOLD > 180:
        errors.append("PHASE_ROTATION_THRESHOLD must be between 0 and 180")
    
    if config.ANALYSIS_WINDOW < 4:
        errors.append("ANALYSIS_WINDOW must be at least 4 samples")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info(f"Found {len(segy_files)} SEG-Y files to process")
    logger.info(f"Configuration validated successfully")
    return True


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Setup logging
    logger = setup_logging()
    
    # Print banner
    print_banner(logger)
    
    # Validate configuration
    if not validate_configuration(logger):
        logger.error("Aborting due to configuration errors")
        return 1
    
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    
    try:
        # Initialize detector
        logger.info("Initializing phase rotation detector...")
        detector = PhaseRotationDetector(
            input_dir=config.INPUT_DIR,
            output_dir=config.OUTPUT_DIR,
            log_level=getattr(logging, config.LOG_LEVEL)
        )
        
        # Initialize QC and export tools
        qc = None
        stats_exporter = None
        
        if config.CREATE_QC_PLOTS or config.CREATE_DETAILED_PLOTS:
            logger.info("Initializing QC tools...")
            qc = PhaseRotationQC(config.OUTPUT_DIR, logger)
        
        if config.EXPORT_STATISTICS:
            logger.info("Initializing statistics exporter...")
            stats_exporter = StatisticsExporter(config.OUTPUT_DIR, logger)
        
        # Get list of files to process
        segy_files = []
        for ext in config.FILE_EXTENSIONS:
            if config.FILENAME_PATTERN:
                segy_files.extend(
                    list(config.INPUT_DIR.glob(f"{config.FILENAME_PATTERN}{ext}"))
                )
            else:
                segy_files.extend(list(config.INPUT_DIR.glob(f"*{ext}")))
        
        logger.info(f"Processing {len(segy_files)} files...")
        
        # Prepare configuration dict
        config_dict = {
            'REFERENCE_TYPE': config.REFERENCE_TYPE,
            'SEAFLOOR_SEARCH_START': config.SEAFLOOR_SEARCH_START,
            'SEAFLOOR_SEARCH_END': config.SEAFLOOR_SEARCH_END,
            'SEAFLOOR_THRESHOLD': config.SEAFLOOR_THRESHOLD,
            'MANUAL_REFERENCE_START': config.MANUAL_REFERENCE_START,
            'MANUAL_REFERENCE_END': config.MANUAL_REFERENCE_END,
            'ANALYSIS_WINDOW': config.ANALYSIS_WINDOW,
            'PHASE_METHOD': config.PHASE_METHOD,
            'SMOOTH_PHASES': config.SMOOTH_PHASES,
            'SMOOTHING_WINDOW': config.SMOOTHING_WINDOW,
            'NORMALIZE_OUTPUT': config.NORMALIZE_OUTPUT,
            'APPLY_GAIN': config.APPLY_GAIN,
            'GAIN_FACTOR': config.GAIN_FACTOR,
            'FREQUENCY_BAND': config.FREQUENCY_BAND,
            'TIME_GATE': config.TIME_GATE,
            'TRACE_DECIMATION': config.TRACE_DECIMATION,
            'REFLECTOR_THRESHOLD_PERCENTILE': config.REFLECTOR_THRESHOLD_PERCENTILE,
            'INDICATOR_TAPER_WIDTH': config.INDICATOR_TAPER_WIDTH
        }
        
        # Storage for results
        results_list = []
        statistics_list = []
        seafloor_picks_dict = {}
        file_output_dirs = {}
        
        # Process files with progress tracking
        from tqdm import tqdm
        
        for filepath in tqdm(segy_files, desc="Processing files", unit="file"):
            try:
                logger.info(f"Processing {filepath.name}")
                
                # Process file
                original_data, phase_rotation, phase_indicator, metadata, seafloor_picks = \
                    detector.process_file(filepath, config_dict)
                
                # Create output directory for this file
                file_stem = filepath.stem
                file_output_dir = config.OUTPUT_DIR / file_stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                file_output_dirs[filepath.name] = file_output_dir
                
                # Decide which attribute to write based on config
                output_type = config.OUTPUT_ATTRIBUTE
                
                if output_type in ['indicator', 'both']:
                    # Write sparse indicator (RECOMMENDED for interpretation)
                    output_segy = file_output_dir / f"{file_stem}_phase_indicator.sgy"
                    detector.write_segy_file(
                        phase_indicator,
                        output_segy,
                        filepath,
                        preserve_headers=config.PRESERVE_INPUT_HEADERS
                    )
                    logger.info(f"Phase indicator attribute saved to {output_segy}")
                
                if output_type in ['full_phase', 'both']:
                    # Write full phase rotation
                    output_segy = file_output_dir / f"{file_stem}_phase_full.sgy"
                    detector.write_segy_file(
                        phase_rotation,
                        output_segy,
                        filepath,
                        preserve_headers=config.PRESERVE_INPUT_HEADERS
                    )
                    logger.info(f"Full phase rotation saved to {output_segy}")
                
                # Create detailed analysis plot
                if config.CREATE_DETAILED_PLOTS and qc:
                    qc_output_dir = file_output_dir / "qc"
                    qc_output_dir.mkdir(parents=True, exist_ok=True)
                    qc_temp = PhaseRotationQC(qc_output_dir, logger)
                    
                    # Use indicator for plotting (more interpretable)
                    qc_temp.create_detailed_plot(
                        original_data,
                        phase_indicator,
                        metadata,
                        seafloor_picks,
                        filepath.name,
                        f"{file_stem}_analysis.png"
                    )
                
                # Collect statistics
                if config.EXPORT_STATISTICS:
                    # Use indicator for statistics
                    phase_data = phase_indicator.flatten()
                    # Only consider non-zero values for meaningful statistics
                    non_zero_phase = phase_data[phase_data != 0]
                    
                    if len(non_zero_phase) > 0:
                        anomalies = np.abs(non_zero_phase) > (config.PHASE_ROTATION_THRESHOLD / 180.0)
                        anomaly_percent = 100 * np.sum(anomalies) / len(non_zero_phase)
                        mean_phase = float(np.mean(non_zero_phase))
                        std_phase = float(np.std(non_zero_phase))
                        min_phase = float(np.min(non_zero_phase))
                        max_phase = float(np.max(non_zero_phase))
                    else:
                        anomaly_percent = 0.0
                        mean_phase = 0.0
                        std_phase = 0.0
                        min_phase = 0.0
                        max_phase = 0.0
                    
                    stats = {
                        'filename': filepath.name,
                        'n_traces': metadata['n_traces'],
                        'n_samples': metadata['n_samples'],
                        'sample_rate': metadata['sample_rate'],
                        'mean_phase': mean_phase,
                        'std_phase': std_phase,
                        'min_phase': min_phase,
                        'max_phase': max_phase,
                        'anomaly_percentage': float(anomaly_percent),
                        'avg_seafloor_time': float(np.mean(seafloor_picks) * metadata['sample_rate'])
                    }
                    statistics_list.append(stats)
                
                # Store seafloor picks
                seafloor_picks_dict[filepath.name] = seafloor_picks
                
                # Store for comparison
                results_list.append((
                    original_data,
                    phase_indicator,  # Use indicator instead of full phase
                    metadata,
                    filepath.name
                ))
                
                logger.info(f"Successfully processed {filepath.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {str(e)}", 
                           exc_info=True)
                continue
        
        # Export statistics
        if config.EXPORT_STATISTICS and stats_exporter and statistics_list:
            logger.info("Exporting statistics...")
            stats_exporter.export_statistics(
                statistics_list,
                config.STATISTICS_CSV_NAME
            )
            
            if seafloor_picks_dict:
                # Get sample rate from first file
                first_sample_rate = statistics_list[0]['sample_rate']
                stats_exporter.export_seafloor_picks(
                    seafloor_picks_dict,
                    first_sample_rate,
                    config.SEAFLOOR_PICKS_NAME
                )
        
        # Create comparison QC plot
        if config.CREATE_QC_PLOTS and qc and results_list:
            logger.info("Creating comparison QC plot...")
            qc.create_comparison_grid(
                results_list[:4],  # Limit to first 4 for readability
                config.QC_PLOT_NAME
            )
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Files processed:        {len(results_list)}/{len(segy_files)}")
        print(f"Total duration:         {duration:.2f} seconds")
        if results_list:
            print(f"Average per file:       {duration/len(results_list):.2f} seconds")
        print(f"\nResults saved to:       {config.OUTPUT_DIR}")
        print("\nGenerated files:")
        
        if config.CREATE_INDIVIDUAL_OUTPUTS:
            output_desc = {
                'indicator': 'phase indicator (sparse)',
                'full_phase': 'full phase rotation (dense)',
                'both': 'both indicator and full phase'
            }
            print(f"  Phase rotation SEG-Y files: {output_desc.get(config.OUTPUT_ATTRIBUTE, 'unknown')}")
            for filename, output_dir in list(file_output_dirs.items())[:3]:
                print(f"    - {output_dir.name}/")
            if len(file_output_dirs) > 3:
                print(f"    ... and {len(file_output_dirs)-3} more")
        
        if config.CREATE_DETAILED_PLOTS:
            print(f"  Detailed analysis plots (in each file's qc/ folder)")
        
        if config.CREATE_QC_PLOTS:
            print(f"  {config.QC_PLOT_NAME} (comparison plot)")
        
        if config.EXPORT_STATISTICS:
            print(f"  {config.STATISTICS_CSV_NAME}")
            print(f"  {config.SEAFLOOR_PICKS_NAME}")
        
        print("="*70)
        
        # Print interpretation guidance
        print("\nInterpretation Guidance:")
        print("  The phase indicator attribute shows phase rotation ONLY at reflectors.")
        print("  ")
        print("  Non-zero values indicate phase-rotated events:")
        print("    - Positive values: Phase advanced relative to reference")
        print("    - Negative values: Phase delayed relative to reference")
        print("    - Larger magnitude = stronger phase rotation")
        print("  ")
        print("  Potential causes of phase rotation:")
        print("    - Hydrocarbon accumulations (polarity reversal)")
        print("    - Fluid contacts (gas-water, oil-water)")
        print("    - Lithology changes (impedance contrasts)")
        print("    - Tuning effects at thin beds")
        print("  ")
        print("  Load the *_phase_indicator.sgy file as an overlay in your")
        print("  interpretation software. Use transparency to see both the")
        print("  original seismic and phase anomalies simultaneously.")
        print("="*70)
        
        logger.info(f"Processing completed successfully in {duration:.2f} seconds")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}", exc_info=True)
        print("\n" + "="*70)
        print("ERROR: Processing failed")
        print(f"Error: {str(e)}")
        print("Check log file for details")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())