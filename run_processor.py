"""
run_processor.py - Main Execution Script
=========================================

Simply run this script after configuring settings in config.py:
    python run_processor.py

All settings are controlled through config.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Import configuration
import config

# Import processing modules
from segy_processor import SEGYProcessor
from segy_qc import SeismicQC, MetadataExporter


def setup_logging():
    """Configure logging based on config settings."""
    # Create output directory first if it doesn't exist
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
    """Print startup banner with configuration summary."""
    banner = f"""
{'='*70}
    SEG-Y Phase Reversal Processing
{'='*70}

Configuration:
  Input Directory:     {config.INPUT_DIR}
  Output Directory:    {config.OUTPUT_DIR}
  
Processing Options:
  Phase Reversal:      {config.PHASE_REVERSAL}
  Normalize:           {config.NORMALIZE_AMPLITUDES}
  Max Workers:         {config.MAX_WORKERS or 'All CPUs'}

Output Options:
  Individual SEG-Y:    {config.CREATE_INDIVIDUAL_SEGY}
  Combined SEG-Y:      {config.CREATE_COMBINED_SEGY}
  Overlay Plot:        {config.CREATE_OVERLAY_PLOT}
  QC Reports:          {config.CREATE_QC_REPORTS}

{'='*70}
    """
    print(banner)
    logger.info("Starting SEG-Y processing")


def validate_configuration(logger):
    """Validate configuration and directories."""
    errors = []
    warnings = []
    
    # Check input directory
    if not config.INPUT_DIR.exists():
        errors.append(f"Input directory does not exist: {config.INPUT_DIR}")
        logger.error(f"Please create the input directory or update INPUT_DIR in config.py")
        return False
    
    # Check for SEG-Y files
    segy_files = []
    for ext in config.FILE_EXTENSIONS:
        if config.FILENAME_PATTERN:
            segy_files.extend(list(config.INPUT_DIR.glob(f"{config.FILENAME_PATTERN}{ext}")))
        else:
            segy_files.extend(list(config.INPUT_DIR.glob(f"*{ext}")))
    
    if not segy_files:
        errors.append(f"No SEG-Y files found in {config.INPUT_DIR}")
        logger.error(f"Please add .sgy or .segy files to {config.INPUT_DIR}")
        return False
    
    # Report errors
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info(f"Found {len(segy_files)} SEG-Y files to process")
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
        # Initialize processor
        logger.info("Initializing SEG-Y processor...")
        processor = SEGYProcessor(
            input_dir=config.INPUT_DIR,
            output_dir=config.OUTPUT_DIR,
            phase_reversed=config.PHASE_REVERSAL,
            normalize=config.NORMALIZE_AMPLITUDES,
            log_level=getattr(logging, config.LOG_LEVEL)
        )
        
        # Initialize QC tools if needed
        qc = None
        metadata_exporter = None
        
        if config.CREATE_QC_REPORTS or config.CREATE_COMPARISON_PLOTS:
            logger.info("Initializing QC tools...")
            qc = SeismicQC(config.OUTPUT_DIR, logger)
        
        if config.EXPORT_METADATA_CSV or config.EXPORT_METADATA_JSON:
            logger.info("Initializing metadata exporter...")
            metadata_exporter = MetadataExporter(config.OUTPUT_DIR, logger)
        
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
        
        # Process files
        processed_data = []
        metadata_list = []
        file_output_dirs = {}  # Track output directories for each file
        
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(processor.process_single_file, f): f 
                for f in segy_files
            }
            
            for future in tqdm(
                as_completed(futures), 
                total=len(futures),
                desc="Processing SEG-Y files",
                unit="file"
            ):
                try:
                    filepath = futures[future]
                    result = future.result()
                    data, metadata, filename = result
                    
                    processed_data.append(result)
                    
                    # Create individual output folder for this file
                    file_stem = Path(filename).stem
                    file_output_dir = config.OUTPUT_DIR / file_stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    file_output_dirs[filename] = file_output_dir
                    
                    # Store metadata
                    metadata['filename'] = filename
                    metadata['output_directory'] = str(file_output_dir)
                    metadata_list.append(metadata)
                    
                    # Create individual output if requested
                    if config.CREATE_INDIVIDUAL_SEGY:
                        output_segy = file_output_dir / f"{file_stem}_phase_reversed.sgy"
                        processor.write_segy_file(
                            data, 
                            metadata, 
                            output_segy,
                            filepath
                        )
                    
                    # Create QC report if requested
                    if config.CREATE_QC_REPORTS and qc:
                        qc_output_dir = file_output_dir / "qc"
                        qc_output_dir.mkdir(parents=True, exist_ok=True)
                        qc_temp = SeismicQC(qc_output_dir, logger)
                        qc_temp.create_comprehensive_qc_plot(
                            data,
                            metadata,
                            filename,
                            f'{file_stem}_qc_report.png'
                        )
                    
                    # Store original data for comparison if needed
                    if config.CREATE_COMPARISON_PLOTS and qc:
                        original_data, _ = processor.read_segy_file(filepath)
                        comparison_output_dir = file_output_dir / "comparison"
                        comparison_output_dir.mkdir(parents=True, exist_ok=True)
                        qc_temp = SeismicQC(comparison_output_dir, logger)
                        qc_temp.create_comparison_plot(
                            original_data,
                            data,
                            metadata,
                            f'{file_stem}_comparison.png'
                        )
                    
                except Exception as e:
                    filepath = futures[future]
                    logger.error(f"Failed to process {filepath.name}: {str(e)}")
        
        # Create overlay plot
        if config.CREATE_OVERLAY_PLOT and processed_data:
            logger.info("Creating overlay plot...")
            overlay_path = config.OUTPUT_DIR / config.OVERLAY_PLOT_NAME
            processor.create_overlay_plot(
                processed_data, 
                overlay_path,
                plot_type=config.PLOT_TYPE,
                figsize=config.FIGURE_SIZE,
                dpi=config.DPI
            )
        
        # Create combined SEG-Y output
        if config.CREATE_COMBINED_SEGY and processed_data:
            logger.info("Creating combined SEG-Y file...")
            combined_output = config.OUTPUT_DIR / config.COMBINED_SEGY_NAME
            
            import numpy as np
            combined_data = np.vstack([d[0] for d in processed_data])
            combined_metadata = processed_data[0][1]
            
            processor.write_segy_file(
                combined_data,
                combined_metadata,
                combined_output,
                segy_files[0]
            )
        
        # Export metadata
        if metadata_exporter:
            if config.EXPORT_METADATA_CSV:
                logger.info("Exporting metadata to CSV...")
                metadata_exporter.export_to_csv(
                    metadata_list, 
                    config.METADATA_CSV_NAME
                )
            
            if config.EXPORT_METADATA_JSON:
                logger.info("Exporting metadata to JSON...")
                metadata_exporter.export_to_json(
                    metadata_list, 
                    config.METADATA_JSON_NAME
                )
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total files processed:  {len(processed_data)}")
        print(f"Total duration:         {duration:.2f} seconds")
        print(f"Average per file:       {duration/len(processed_data):.2f} seconds")
        print(f"\nResults saved to:       {config.OUTPUT_DIR}")
        print("\nGenerated files:")
        
        if config.CREATE_INDIVIDUAL_SEGY:
            print(f"{len(processed_data)} individual output folders with processed SEG-Y files")
            for filename, output_dir in list(file_output_dirs.items())[:3]:
                print(f"    - {output_dir.name}/")
            if len(file_output_dirs) > 3:
                print(f"    ... and {len(file_output_dirs)-3} more")
        
        if config.CREATE_COMBINED_SEGY:
            print(f"{config.COMBINED_SEGY_NAME} (in main output folder)")
        if config.CREATE_OVERLAY_PLOT:
            print(f"{config.OVERLAY_PLOT_NAME} (in main output folder)")
        if config.CREATE_QC_REPORTS:
            print(f"QC reports in each file's folder (qc/ subfolder)")
        if config.CREATE_COMPARISON_PLOTS:
            print(f"Comparison plots in each file's folder (comparison/ subfolder)")
        if config.EXPORT_METADATA_CSV:
            print(f"{config.METADATA_CSV_NAME} (in main output folder)")
        if config.EXPORT_METADATA_JSON:
            print(f"{config.METADATA_JSON_NAME} (in main output folder)")
        
        print("="*70)
        
        logger.info(f"Processing completed successfully in {duration:.2f} seconds")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}", exc_info=True)
        print("\n" + "="*70)
        print("ERROR: Processing failed!")
        print(f"Error: {str(e)}")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())