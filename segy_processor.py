"""
segy_processor.py - Core SEG-Y Processing Module
=================================================

Handles reading, processing, and writing of SEG-Y files with
phase reversal and overlay plotting capabilities.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import segyio
import warnings

warnings.filterwarnings('ignore')


class SEGYProcessor:
    """
    Main processor for SEG-Y files with phase reversal.
    
    Handles all core functionality for reading, processing,
    and writing seismic data files.
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        phase_reversed: bool = True,
        normalize: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize the SEG-Y processor.
        
        Args:
            input_dir: Directory containing input SEG-Y files
            output_dir: Directory for output files
            phase_reversed: Apply 180° phase reversal
            normalize: Normalize amplitudes to [-1, 1]
            log_level: Logging level
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.phase_reversed = phase_reversed
        self.normalize = normalize
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def read_segy_file(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        Read SEG-Y file and extract trace data with headers.
        
        Args:
            filepath: Path to SEG-Y file
            
        Returns:
            Tuple of (trace_data, metadata_dict)
        """
        try:
            with segyio.open(str(filepath), 'r', ignore_geometry=True) as segy:
                # Extract all traces
                traces = segyio.collect(segy.trace[:])
                
                # Extract metadata - convert binary header to JSON-serializable format
                binary_header = {}
                for key, value in segy.bin.items():
                    try:
                        # Convert BinField keys to strings
                        binary_header[str(key)] = int(value) if isinstance(value, (int, np.integer)) else value
                    except:
                        binary_header[str(key)] = str(value)
                
                metadata = {
                    'sample_rate': segyio.tools.dt(segy) / 1000.0,
                    'n_traces': segy.tracecount,
                    'n_samples': segy.samples.size,
                    'text_header': segy.text[0].decode('ascii', errors='ignore'),
                    'binary_header': binary_header,
                    'trace_headers': []
                }
                
                # Extract limited trace headers for performance
                n_headers = min(segy.tracecount, 100)
                for i in range(n_headers):
                    header = segy.header[i]
                    metadata['trace_headers'].append({
                        'inline': int(header[segyio.TraceField.INLINE_3D]),
                        'crossline': int(header[segyio.TraceField.CROSSLINE_3D]),
                        'cdp_x': int(header[segyio.TraceField.CDP_X]),
                        'cdp_y': int(header[segyio.TraceField.CDP_Y]),
                        'trace_number': int(header[segyio.TraceField.TraceNumber])
                    })
                
                self.logger.info(
                    f"Read {filepath.name}: {metadata['n_traces']} traces, "
                    f"{metadata['n_samples']} samples"
                )
                
                return traces, metadata
                
        except Exception as e:
            self.logger.error(f"Failed to read {filepath}: {str(e)}")
            raise IOError(f"Cannot read SEG-Y file: {filepath}") from e
    
    def apply_phase_reversal(self, data: np.ndarray) -> np.ndarray:
        """
        Apply 180° phase reversal to seismic data.
        
        Args:
            data: Input seismic trace data
            
        Returns:
            Phase-reversed data
        """
        if not self.phase_reversed:
            return data
            
        self.logger.debug("Applying phase reversal")
        return -data
    
    def normalize_traces(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize trace amplitudes to [-1, 1] range.
        
        Args:
            data: Input seismic trace data
            
        Returns:
            Normalized data
        """
        if not self.normalize:
            return data
            
        self.logger.debug("Normalizing amplitudes")
        max_amp = np.abs(data).max()
        if max_amp > 0:
            return data / max_amp
        return data
    
    def create_overlay_plot(
        self,
        data_list: List[Tuple[np.ndarray, Dict, str]],
        output_path: Path,
        plot_type: str = 'wiggle',
        figsize: Tuple[int, int] = (16, 10),
        dpi: int = 300
    ) -> None:
        """
        Create overlay plot of multiple SEG-Y datasets.
        
        Args:
            data_list: List of tuples (data, metadata, filename)
            output_path: Path for output PNG file
            plot_type: 'wiggle', 'density', or 'both'
            figsize: Figure size in inches
            dpi: Resolution
        """
        self.logger.info(f"Creating overlay plot with {len(data_list)} datasets")
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
        
        for idx, (data, metadata, filename) in enumerate(data_list):
            sample_rate = metadata['sample_rate']
            n_samples = data.shape[1]
            time_axis = np.arange(n_samples) * sample_rate
            
            # Calculate mean trace
            mean_trace = np.mean(data, axis=0)
            
            if plot_type in ['wiggle', 'both']:
                ax.plot(
                    mean_trace, 
                    time_axis,
                    color=colors[idx],
                    linewidth=0.8,
                    label=Path(filename).stem,
                    alpha=0.7
                )
            
            if plot_type in ['density', 'both']:
                extent = [mean_trace.min(), mean_trace.max(),
                         time_axis.max(), time_axis.min()]
                ax.imshow(data.T, aspect='auto', cmap='seismic',
                         alpha=0.3, extent=extent)
        
        ax.set_xlabel('Amplitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Phase Reversed Seismic Overlay', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Overlay plot saved to {output_path}")
    
    def write_segy_file(
        self,
        data: np.ndarray,
        metadata: Dict,
        output_path: Path,
        reference_file: Optional[Path] = None
    ) -> None:
        """
        Write processed data to SEG-Y file with header preservation.
        
        Args:
            data: Processed seismic trace data
            metadata: Metadata dictionary
            output_path: Path for output SEG-Y file
            reference_file: Reference file for header copying
        """
        try:
            spec = segyio.spec()
            spec.format = 1
            spec.samples = np.arange(data.shape[1])
            spec.tracecount = data.shape[0]
            
            with segyio.create(str(output_path), spec) as dst:
                # Copy headers from reference if available
                if reference_file and reference_file.exists():
                    with segyio.open(str(reference_file), 'r', 
                                   ignore_geometry=True) as src:
                        dst.text[0] = src.text[0]
                        dst.bin = src.bin
                        
                        # Copy trace headers
                        for i in range(min(spec.tracecount, src.tracecount)):
                            dst.header[i] = src.header[i]
                else:
                    # Set basic binary header
                    dst.bin[segyio.BinField.Samples] = data.shape[1]
                    dst.bin[segyio.BinField.Interval] = int(
                        metadata['sample_rate'] * 1000
                    )
                
                # Write trace data
                for i, trace in enumerate(data):
                    dst.trace[i] = trace
            
            self.logger.info(f"SEG-Y file written to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write SEG-Y: {str(e)}")
            raise IOError(f"Cannot write SEG-Y file: {output_path}") from e
    
    def process_single_file(self, filepath: Path) -> Tuple[np.ndarray, Dict, str]:
        """
        Process a single SEG-Y file.
        
        Args:
            filepath: Path to input SEG-Y file
            
        Returns:
            Tuple of (processed_data, metadata, filename)
        """
        try:
            data, metadata = self.read_segy_file(filepath)
            data = self.apply_phase_reversal(data)
            data = self.normalize_traces(data)
            return data, metadata, filepath.name
        except Exception as e:
            self.logger.error(f"Error processing {filepath.name}: {str(e)}")
            raise