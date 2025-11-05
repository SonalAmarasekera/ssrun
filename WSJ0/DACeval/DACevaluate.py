#!/usr/bin/env python
"""
Compare original audio files with encoded-decoded files using objective metrics.
Uses torchmetrics for PESQ, SI-SDR, STOI, and SNR.
"""

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any
import os
import sys

import torch
import tqdm
import numpy as np
from audiotools.core import AudioSignal

# Import torchmetrics
try:
    from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalDistortionRatio, ShortTimeObjectiveIntelligibility
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    TORCHMETRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: torchmetrics not available: {e}")
    print("Please install with: pip install torchmetrics[audio]")
    TORCHMETRICS_AVAILABLE = False


def get_metric_function(metric_name: str, device: str = 'cpu'):
    """Get the appropriate metric function based on name."""
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError("torchmetrics is required for this script")
    
    metric_functions = {}
    
    if metric_name.lower() == "snr":
        metric = SignalNoiseRatio().to(device)
        metric_functions["snr"] = lambda x, y: metric(
            torch.tensor(y.audio_data, device=device).unsqueeze(0) if len(y.audio_data.shape) == 1 else torch.tensor(y.audio_data, device=device),
            torch.tensor(x.audio_data, device=device).unsqueeze(0) if len(x.audio_data.shape) == 1 else torch.tensor(x.audio_data, device=device)
        )
    
    elif metric_name.lower() == "sisdr":
        metric = ScaleInvariantSignalDistortionRatio().to(device)
        metric_functions["sisdr"] = lambda x, y: metric(
            torch.tensor(y.audio_data, device=device).unsqueeze(0) if len(y.audio_data.shape) == 1 else torch.tensor(y.audio_data, device=device),
            torch.tensor(x.audio_data, device=device).unsqueeze(0) if len(x.audio_data.shape) == 1 else torch.tensor(x.audio_data, device=device)
        )
    
    elif metric_name.lower() == "stoi":
        metric_functions["stoi"] = lambda x, y: ShortTimeObjectiveIntelligibility(
            x.sample_rate, extended=False
        ).to(device)(
            torch.tensor(y.audio_data, device=device).unsqueeze(0) if len(y.audio_data.shape) == 1 else torch.tensor(y.audio_data, device=device),
            torch.tensor(x.audio_data, device=device).unsqueeze(0) if len(x.audio_data.shape) == 1 else torch.tensor(x.audio_data, device=device)
        )
    
    elif metric_name.lower() == "pesq":
        # PESQ requires specific sample rates
        def pesq_wrapper(x, y):
            try:
                # PESQ only supports 16000 and 8000 Hz
                if x.sample_rate not in [8000, 16000]:
                    # Resample to 16000 for PESQ
                    x_resampled = x.clone().resample(16000)
                    y_resampled = y.clone().resample(16000)
                    pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
                else:
                    x_resampled = x
                    y_resampled = y
                    mode = 'nb' if x.sample_rate == 8000 else 'wb'
                    pesq = PerceptualEvaluationSpeechQuality(x.sample_rate, mode)
                
                return pesq.to(device)(
                    torch.tensor(y_resampled.audio_data, device=device).unsqueeze(0) if len(y_resampled.audio_data.shape) == 1 else torch.tensor(y_resampled.audio_data, device=device),
                    torch.tensor(x_resampled.audio_data, device=device).unsqueeze(0) if len(x_resampled.audio_data.shape) == 1 else torch.tensor(x_resampled.audio_data, device=device)
                )
            except Exception as e:
                print(f"PESQ error: {e}")
                return torch.tensor(float('nan'))
        
        metric_functions["pesq"] = pesq_wrapper
    
    return metric_functions.get(metric_name.lower())


def compute_metrics_for_pair(args):
    """Compute metrics for a single original-processed file pair."""
    original_path, processed_path, metric_names, sample_rates, device = args
    
    try:
        # Load both audio files
        original = AudioSignal(original_path)
        processed = AudioSignal(processed_path)
        
        results = {
            "original_path": str(original_path),
            "processed_path": str(processed_path),
            "original_duration": original.duration,
            "processed_duration": processed.duration,
            "original_sample_rate": original.sample_rate,
            "processed_sample_rate": processed.sample_rate,
        }
        
        # Compute metrics at different sample rates if requested
        for sr in sample_rates:
            # Skip PESQ for sample rates it doesn't support (we handle PESQ separately)
            current_metrics = [m for m in metric_names if m != 'pesq' or sr in [8000, 16000]]
            
            if sr != original.sample_rate:
                orig_resampled = original.clone().resample(sr)
                proc_resampled = processed.clone().resample(sr)
            else:
                orig_resampled = original
                proc_resampled = processed
            
            sr_key = f"{sr}hz"
            
            for metric_name in current_metrics:
                try:
                    metric_fn = get_metric_function(metric_name, device)
                    if metric_fn is None:
                        results[f"{metric_name}_{sr_key}"] = float('nan')
                        continue
                    
                    metric_value = metric_fn(orig_resampled, proc_resampled)
                    if torch.is_tensor(metric_value):
                        metric_value = metric_value.cpu().item()
                    results[f"{metric_name}_{sr_key}"] = float(metric_value)
                except Exception as e:
                    print(f"Error computing {metric_name} at {sr}Hz for {original_path}: {e}")
                    results[f"{metric_name}_{sr_key}"] = float('nan')
        
        # Handle PESQ separately since it has specific sample rate requirements
        if 'pesq' in metric_names:
            try:
                pesq_fn = get_metric_function('pesq', device)
                if pesq_fn:
                    # PESQ will handle resampling internally if needed
                    pesq_value = pesq_fn(original, processed)
                    if torch.is_tensor(pesq_value):
                        pesq_value = pesq_value.cpu().item()
                    results["pesq"] = float(pesq_value)
            except Exception as e:
                print(f"Error computing PESQ for {original_path}: {e}")
                results["pesq"] = float('nan')
        
        return results
        
    except Exception as e:
        print(f"Error processing {original_path}: {e}")
        return {
            "original_path": str(original_path),
            "processed_path": str(processed_path),
            "error": str(e)
        }


def find_matching_files(original_dir: Path, processed_dir: Path) -> List[tuple]:
    """Find matching file pairs between original and processed directories."""
    original_files = []
    processed_files = []
    
    # Find all audio files in original directory
    audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.aac'}
    for ext in audio_extensions:
        original_files.extend(original_dir.rglob(f"*{ext}"))
        original_files.extend(original_dir.rglob(f"*{ext.upper()}"))
    
    # Find corresponding files in processed directory
    matching_pairs = []
    for orig_file in original_files:
        # Get relative path from original directory
        if original_dir.is_file():
            rel_path = Path(orig_file.name)
        else:
            try:
                rel_path = orig_file.relative_to(original_dir)
            except ValueError:
                # If files are not in a subdirectory, use filename only
                rel_path = Path(orig_file.name)
        
        # Look for processed file with same relative path
        processed_file = processed_dir / rel_path
        
        # Try different extensions if needed
        if not processed_file.exists():
            # Try with .wav extension (common output format)
            processed_file = processed_file.with_suffix('.wav')
        
        if processed_file.exists():
            matching_pairs.append((orig_file, processed_file))
        else:
            print(f"Warning: No matching processed file for {orig_file}")
    
    return matching_pairs


def compute_summary_statistics(results: List[Dict], metric_names: List[str], sample_rates: List[int]) -> Dict[str, Dict]:
    """Compute summary statistics for all metrics."""
    summary = {}
    
    for metric in metric_names:
        summary[metric] = {}
        
        # Handle PESQ separately (no sample rate suffix)
        if metric == 'pesq':
            metric_values = []
            for result in results:
                if 'pesq' in result and isinstance(result['pesq'], (int, float)) and not np.isnan(result['pesq']):
                    metric_values.append(result['pesq'])
            
            if metric_values:
                summary[metric]['all'] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'count': len(metric_values)
                }
            continue
        
        # Handle other metrics with sample rates
        for sr in sample_rates:
            sr_key = f"{sr}hz"
            metric_values = []
            
            for result in results:
                key = f"{metric}_{sr_key}"
                if key in result and isinstance(result[key], (int, float)) and not np.isnan(result[key]):
                    metric_values.append(result[key])
            
            if metric_values:
                summary[metric][sr_key] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'count': len(metric_values)
                }
            else:
                summary[metric][sr_key] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'count': 0
                }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and processed audio files using torchmetrics"
    )
    
    # Required arguments
    parser.add_argument("--input_dir", required=True, 
                       help="Directory containing original audio files")
    parser.add_argument("--output_dir", required=True,
                       help="Directory containing processed (encoded-decoded) audio files")
    
    # Metric selection
    parser.add_argument("--metrics", nargs="+", required=True,
                       choices=["snr", "sisdr", "stoi", "pesq"],
                       help="Metrics to compute (from torchmetrics)")
    
    # Performance settings
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2),
                       help="Number of worker processes")
    parser.add_argument("--sample_rates", nargs="+", type=int, default=[16000, 22050, 44100],
                       help="Sample rates to compute metrics at (except PESQ)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                       help="Device to run metrics on")
    
    # Output options
    parser.add_argument("--results_file", default="torchmetrics_comparison.csv",
                       help="Output CSV file for results")
    parser.add_argument("--summary_file", default="torchmetrics_summary.txt",
                       help="Output text file with summary statistics")
    
    args = parser.parse_args()
    
    if not TORCHMETRICS_AVAILABLE:
        print("Error: torchmetrics is required but not available.")
        print("Please install with: pip install torchmetrics[audio]")
        sys.exit(1)
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    results_file = Path(args.results_file)
    summary_file = Path(args.summary_file)
    
    # Validate directories
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    print("=== Audio Quality Comparison (torchmetrics) ===")
    print(f"Original files: {input_dir}")
    print(f"Processed files: {output_dir}")
    print(f"Metrics: {', '.join(args.metrics)}")
    print(f"Sample rates: {args.sample_rates}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print()
    
    # Find matching file pairs
    print("Finding matching audio files...")
    file_pairs = find_matching_files(input_dir, output_dir)
    
    if not file_pairs:
        raise ValueError("No matching file pairs found between input and output directories")
    
    print(f"Found {len(file_pairs)} matching file pairs")
    print()
    
    # Prepare arguments for multiprocessing
    mp_args = [
        (orig, proc, args.metrics, args.sample_rates, args.device)
        for orig, proc in file_pairs
    ]
    
    # Compute metrics using multiprocessing
    print("Computing metrics...")
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        # Submit all jobs
        futures = [executor.submit(compute_metrics_for_pair, arg) for arg in mp_args]
        
        # Process results with progress bar
        for future in tqdm.tqdm(futures, total=len(mp_args), desc="Processing files"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error processing future: {e}")
                all_results.append({"error": str(e)})
    
    # Filter out error results for summary
    valid_results = [r for r in all_results if "error" not in r]
    
    print(f"\nSuccessfully processed {len(valid_results)} out of {len(all_results)} files")
    
    # Write detailed results to CSV
    print(f"Writing detailed results to {results_file}...")
    if all_results:
        # Get all possible fieldnames
        fieldnames = set()
        for result in all_results:
            fieldnames.update(result.keys())
        fieldnames = sorted(fieldnames)
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
    
    # Compute and display summary statistics
    print(f"Computing summary statistics...")
    summary = compute_summary_statistics(valid_results, args.metrics, args.sample_rates)
    
    # Write summary to file
    with open(summary_file, 'w') as f:
        f.write("=== Audio Quality Comparison Summary (torchmetrics) ===\n")
        f.write(f"Original files: {input_dir}\n")
        f.write(f"Processed files: {output_dir}\n")
        f.write(f"Total files compared: {len(valid_results)}/{len(all_results)}\n")
        f.write(f"Metrics: {', '.join(args.metrics)}\n\n")
        
        for metric in args.metrics:
            f.write(f"--- {metric.upper()} ---\n")
            
            if metric == 'pesq':
                if 'all' in summary[metric]:
                    stats = summary[metric]['all']
                    f.write(f"  PESQ: mean={stats['mean']:.6f} ± {stats['std']:.6f}, "
                           f"range=[{stats['min']:.6f}, {stats['max']:.6f}], n={stats['count']}\n")
                else:
                    f.write(f"  PESQ: No valid values\n")
            else:
                for sr in args.sample_rates:
                    sr_key = f"{sr}hz"
                    if sr_key in summary[metric]:
                        stats = summary[metric][sr_key]
                        if stats['count'] > 0:
                            f.write(f"  {sr}Hz: mean={stats['mean']:.6f} ± {stats['std']:.6f}, "
                                   f"range=[{stats['min']:.6f}, {stats['max']:.6f}], n={stats['count']}\n")
                        else:
                            f.write(f"  {sr}Hz: No valid values\n")
            f.write("\n")
    
    # Print quick summary to console
    print("\n=== Quick Summary ===")
    for metric in args.metrics:
        print(f"\n{metric.upper()}:")
        
        if metric == 'pesq':
            if 'all' in summary[metric]:
                stats = summary[metric]['all']
                print(f"  PESQ: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")
        else:
            for sr in args.sample_rates:
                sr_key = f"{sr}hz"
                if sr_key in summary[metric]:
                    stats = summary[metric][sr_key]
                    if stats['count'] > 0:
                        print(f"  {sr}Hz: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")
    
    print(f"\nDetailed results: {results_file}")
    print(f"Summary statistics: {summary_file}")


if __name__ == "__main__":
    main()