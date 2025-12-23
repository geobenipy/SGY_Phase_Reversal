"""
segy_qc.py - Quality Control and Metadata Export
=================================================

Quality control tools and metadata export functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional
import logging
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
import seaborn as sns

sns.set_style("whitegrid")


class SeismicQC:
    """Quality control and visualization tools."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        """Initialize QC module."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_statistics(self, data: np.ndarray) -> Dict:
        """Calculate comprehensive statistics."""
        stats_dict = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'rms': np.sqrt(np.mean(data**2)),
            'skewness': stats.skew(data.flatten()),
            'kurtosis': stats.kurtosis(data.flatten()),
            'percentile_95': np.percentile(data, 95),
            'percentile_05': np.percentile(data, 5)
        }
        
        self.logger.info(
            f"Stats - RMS: {stats_dict['rms']:.4f}, "
            f"Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]"
        )
        
        return stats_dict
    
    def frequency_spectrum(
        self,
        data: np.ndarray,
        sample_rate: float,
        trace_index: int = 0
    ) -> tuple:
        """Compute frequency spectrum."""
        trace = data[trace_index]
        n_samples = len(trace)
        
        spectrum = fft(trace)
        freqs = fftfreq(n_samples, d=sample_rate/1000.0)
        
        positive_freqs = freqs[:n_samples//2]
        amplitudes = np.abs(spectrum[:n_samples//2])
        
        return positive_freqs, amplitudes
    
    def create_comprehensive_qc_plot(
        self,
        data: np.ndarray,
        metadata: Dict,
        filename: str,
        output_name: str = 'qc_report.png'
    ) -> None:
        """Create comprehensive QC plot with multiple panels."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Seismic section
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_seismic_section(ax1, data, metadata)
        ax1.set_title(f'Seismic Section: {filename}', 
                     fontweight='bold', fontsize=14)
        
        # Amplitude histogram
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_amplitude_histogram(ax2, data)
        
        # Frequency spectrum
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_frequency_spectrum(ax3, data, metadata['sample_rate'])
        
        # Statistics table
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_statistics_table(ax4, data)
        
        # Trace comparison
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_trace_comparison(ax5, data, metadata['sample_rate'])
        
        # RMS evolution
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_rms_evolution(ax6, data)
        
        # Phase analysis
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_phase_analysis(ax7, data)
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"QC report saved to {output_path}")
    
    def _plot_seismic_section(self, ax, data, metadata):
        """Plot seismic section."""
        sample_rate = metadata['sample_rate']
        n_traces, n_samples = data.shape
        extent = [0, n_traces, n_samples * sample_rate, 0]
        
        im = ax.imshow(
            data.T, aspect='auto', cmap='seismic',
            extent=extent, interpolation='bilinear',
            vmin=-np.percentile(np.abs(data), 99),
            vmax=np.percentile(np.abs(data), 99)
        )
        
        ax.set_xlabel('Trace Number', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Amplitude')
    
    def _plot_amplitude_histogram(self, ax, data):
        """Plot amplitude distribution."""
        ax.hist(data.flatten(), bins=100, color='steelblue',
               edgecolor='black', alpha=0.7)
        ax.set_xlabel('Amplitude', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Amplitude Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    def _plot_frequency_spectrum(self, ax, data, sample_rate):
        """Plot frequency spectrum."""
        n_traces_to_avg = min(10, data.shape[0])
        freqs, amps = self.frequency_spectrum(data, sample_rate, 0)
        avg_spectrum = np.zeros_like(amps)
        
        for i in range(n_traces_to_avg):
            _, spec = self.frequency_spectrum(data, sample_rate, i)
            avg_spectrum += spec
        
        avg_spectrum /= n_traces_to_avg
        
        ax.plot(freqs, avg_spectrum, color='darkblue', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title('Average Frequency Spectrum', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(250, freqs.max())])
    
    def _plot_statistics_table(self, ax, data):
        """Plot statistics table."""
        stats = self.calculate_statistics(data)
        
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [
            ['Statistic', 'Value'],
            ['Mean', f"{stats['mean']:.4f}"],
            ['Std Dev', f"{stats['std']:.4f}"],
            ['Min', f"{stats['min']:.4f}"],
            ['Max', f"{stats['max']:.4f}"],
            ['RMS', f"{stats['rms']:.4f}"],
            ['Skewness', f"{stats['skewness']:.4f}"],
            ['Kurtosis', f"{stats['kurtosis']:.4f}"]
        ]
        
        table = ax.table(cellText=table_data, cellLoc='left',
                        loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    def _plot_trace_comparison(self, ax, data, sample_rate):
        """Plot trace comparison."""
        n_traces = data.shape[0]
        time_axis = np.arange(data.shape[1]) * sample_rate
        
        traces_to_plot = [0, n_traces//2, n_traces-1]
        labels = ['First', 'Middle', 'Last']
        colors = ['blue', 'green', 'red']
        
        for idx, label, color in zip(traces_to_plot, labels, colors):
            ax.plot(data[idx], time_axis, label=f'{label} Trace',
                   color=color, linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Amplitude', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Trace Comparison', fontweight='bold')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rms_evolution(self, ax, data):
        """Plot RMS amplitude evolution."""
        rms_values = np.sqrt(np.mean(data**2, axis=1))
        
        ax.plot(rms_values, color='darkred', linewidth=1.5)
        ax.set_xlabel('Trace Number', fontweight='bold')
        ax.set_ylabel('RMS Amplitude', fontweight='bold')
        ax.set_title('RMS Amplitude Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        z = np.polyfit(range(len(rms_values)), rms_values, 1)
        p = np.poly1d(z)
        ax.plot(range(len(rms_values)), p(range(len(rms_values))),
               "r--", alpha=0.5, label='Trend')
        ax.legend()
    
    def _plot_phase_analysis(self, ax, data):
        """Plot phase analysis."""
        trace = data[data.shape[0]//2]
        analytic_signal = hilbert(trace)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        time_axis = np.arange(len(trace))
        
        ax2 = ax.twinx()
        ax.plot(time_axis, trace, 'b-', linewidth=1, 
               label='Amplitude', alpha=0.7)
        ax2.plot(time_axis, instantaneous_phase, 'r-',
                linewidth=1, label='Phase', alpha=0.7)
        
        ax.set_xlabel('Sample Number', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold', color='b')
        ax2.set_ylabel('Phase (radians)', fontweight='bold', color='r')
        ax.set_title('Phase Analysis', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    def create_comparison_plot(
        self,
        original_data: np.ndarray,
        processed_data: np.ndarray,
        metadata: Dict,
        output_name: str = 'comparison.png'
    ) -> None:
        """Create before/after comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        sample_rate = metadata['sample_rate']
        n_traces, n_samples = original_data.shape
        extent = [0, n_traces, n_samples * sample_rate, 0]
        
        vmax = np.percentile(np.abs(original_data), 99)
        
        im1 = ax1.imshow(original_data.T, aspect='auto', cmap='seismic',
                        extent=extent, vmin=-vmax, vmax=vmax)
        ax1.set_title('Original Data', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Trace Number', fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Amplitude')
        
        im2 = ax2.imshow(processed_data.T, aspect='auto', cmap='seismic',
                        extent=extent, vmin=-vmax, vmax=vmax)
        ax2.set_title('Phase Reversed Data', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Trace Number', fontweight='bold')
        ax2.set_ylabel('Time (ms)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Amplitude')
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved to {output_path}")


class MetadataExporter:
    """Export seismic metadata to various formats."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        """Initialize metadata exporter."""
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
    
    def export_to_csv(self, metadata_list: List[Dict], filename: str):
        """Export metadata to CSV."""
        import csv
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as csvfile:
            if not metadata_list:
                return
            
            fieldnames = ['file', 'n_traces', 'n_samples', 'sample_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for meta in metadata_list:
                writer.writerow({
                    'file': meta.get('filename', 'unknown'),
                    'n_traces': meta.get('n_traces', 0),
                    'n_samples': meta.get('n_samples', 0),
                    'sample_rate': meta.get('sample_rate', 0)
                })
        
        self.logger.info(f"Metadata exported to {output_path}")
    
    def export_to_json(self, metadata_list: List[Dict], filename: str):
        """Export metadata to JSON."""
        import json
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as jsonfile:
            json.dump(metadata_list, jsonfile, indent=2, default=str)
        
        self.logger.info(f"Metadata exported to {output_path}")