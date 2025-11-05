#!/usr/bin/env python
"""
Ultra-fast parallel encoding and decoding with Descript Audio Codec.
Uses the same optimized multiprocessing approach for both phases.
"""

import argparse
import os
import pathlib
import sys
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict
import time

import numpy as np
import soundfile as sf
import torch
import tqdm

import dac
from audiotools import AudioSignal


# ---------------------------- Configuration ----------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MODEL_CACHE = {}  # Global model cache to avoid repeated downloads


# ---------------------------- Model Management ----------------------------

def get_model(model_type: str, device: str):
    """Get model from cache or download if not present."""
    cache_key = f"{model_type}_{device}"
    if cache_key not in MODEL_CACHE:
        model_path = dac.utils.download(model_type=model_type)
        model = dac.DAC.load(model_path)
        model.to(device)
        MODEL_CACHE[cache_key] = model
    return MODEL_CACHE[cache_key]


def preload_models(model_type: str, device: str, num_workers: int):
    """Preload models to warm up the cache."""
    print("Preloading models...")
    for i in range(min(num_workers, 4)):  # Preload a few models
        get_model(model_type, device)


# ---------------------------- File Discovery ----------------------------

def find_audio_paths(root: pathlib.Path) -> List[pathlib.Path]:
    """Find all audio files in directory."""
    if root.is_file():
        return [root]
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            files.append(p)
    return files


def find_dac_files(temp_dir: pathlib.Path) -> List[pathlib.Path]:
    """Find all .dac files in the temporary directory."""
    dac_files = []
    for p in temp_dir.rglob("*.dac"):
        if p.is_file():
            dac_files.append(p)
    dac_files.sort()
    return dac_files


# ---------------------------- Encoding Workers ----------------------------

def _worker_encode(args):
    """Worker for encoding: convert audio to .dac format."""
    audio_path, dac_path, model_type, device, expect_sr = args
    
    if dac_path.exists():
        return audio_path, dac_path, "skipped"
    
    try:
        # Get model (will use cached version if available)
        model = get_model(model_type, device)
        
        # Read audio file
        wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if wav.ndim == 2:  # mixdown if stereo
            wav = wav.mean(axis=1)
        
        if expect_sr is not None and sr != expect_sr:
            # Resample if needed
            signal = AudioSignal(audio_path)
            signal = signal.resample(expect_sr)
        else:
            signal = AudioSignal(audio_path)
        
        signal = signal.cpu()
        compressed = model.compress(signal)
        
        # Save DAC file
        dac_path.parent.mkdir(parents=True, exist_ok=True)
        compressed.save(dac_path)
        
        return audio_path, dac_path, "success"
        
    except Exception as e:
        print(f"Error encoding {audio_path}: {str(e)}")
        return audio_path, dac_path, f"error: {str(e)}"


# ---------------------------- Decoding Workers ----------------------------

def _worker_decode(args):
    """Worker for decoding: convert .dac format back to WAV."""
    dac_path, output_path, model_type, device, expect_sr = args
    
    if output_path.exists():
        return dac_path, output_path, "skipped"
    
    try:
        # Get model (will use cached version if available)
        model = get_model(model_type, device)
        
        # Load and decompress DAC file
        compressed = dac.DACFile.load(dac_path)
        decompressed = model.decompress(compressed)
        
        # Create output directory and write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decompressed.write(output_path)
        
        return dac_path, output_path, "success"
        
    except Exception as e:
        print(f"Error decoding {dac_path}: {str(e)}")
        return dac_path, output_path, f"error: {str(e)}"


# ---------------------------- Parallel Processing Engine ----------------------------

def parallel_process(jobs, worker_function, desc: str, workers: int, device: str):
    """Generic parallel processing function for both encoding and decoding."""
    if workers <= 1:
        # Sequential processing
        results = []
        for job in tqdm.tqdm(jobs, desc=desc):
            results.append(worker_function(job))
        return results
    
    # Parallel processing with optimal chunking
    ctx = mp.get_context("spawn")
    
    # Adjust chunk size based on job count and workers
    chunk_size = max(1, len(jobs) // (workers * 10))
    
    with ctx.Pool(
        processes=workers,
        maxtasksperchild=50,  # Clean up workers periodically to free memory
    ) as pool:
        results = list(tqdm.tqdm(
            pool.imap(worker_function, jobs, chunksize=chunk_size),
            total=len(jobs),
            desc=desc
        ))
    
    # Clear GPU cache after parallel processing
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return results


# ---------------------------- Encoding Phase ----------------------------

def encode_phase(input_dir: pathlib.Path, temp_dir: pathlib.Path, 
                model_type: str, device: str, workers: int):
    """Fast encoding phase using multiprocessing."""
    print("=== ENCODING PHASE ===")
    
    # Get expected sample rate
    expect_sr = {"16khz": 16000, "24khz": 24000, "44khz": 44100}[model_type]
    
    # Find all audio files
    audio_files = find_audio_paths(input_dir)
    audio_files.sort()
    print(f"Found {len(audio_files)} audio files to encode")
    
    # Prepare encoding jobs
    encoding_jobs = []
    for audio_path in audio_files:
        rel_path = audio_path.relative_to(input_dir) if input_dir.is_dir() else pathlib.Path(audio_path.name)
        dac_path = (temp_dir / rel_path).with_suffix(".dac")
        encoding_jobs.append((audio_path, dac_path, model_type, device, expect_sr))
    
    # Process in parallel
    start_time = time.time()
    results = parallel_process(encoding_jobs, _worker_encode, "Encoding", workers, device)
    encoding_time = time.time() - start_time
    
    # Print statistics
    success_count = sum(1 for r in results if r[2] == "success")
    skipped_count = sum(1 for r in results if r[2] == "skipped")
    error_count = sum(1 for r in results if r[2].startswith("error"))
    
    print(f"Encoding complete: {success_count} success, {skipped_count} skipped, {error_count} errors")
    print(f"Encoding time: {encoding_time:.2f}s ({len(audio_files)/encoding_time:.2f} files/sec)")
    print(f"DAC files saved to: {temp_dir}")
    
    return [job[1] for job in encoding_jobs]


# ---------------------------- Decoding Phase ----------------------------

def decode_phase(temp_dir: pathlib.Path, output_dir: pathlib.Path, 
                model_type: str, device: str, workers: int):
    """Fast decoding phase using the same parallel approach as encoding."""
    print("\n=== DECODING PHASE ===")
    
    # Find all DAC files
    dac_files = find_dac_files(temp_dir)
    print(f"Found {len(dac_files)} DAC files to decode")
    
    # Get expected sample rate
    expect_sr = {"16khz": 16000, "24khz": 24000, "44khz": 44100}[model_type]
    
    # Prepare decoding jobs
    decoding_jobs = []
    for dac_path in dac_files:
        rel_path = dac_path.relative_to(temp_dir)
        output_path = (output_dir / rel_path).with_suffix(".wav")
        decoding_jobs.append((dac_path, output_path, model_type, device, expect_sr))
    
    # Process in parallel (same function as encoding)
    start_time = time.time()
    results = parallel_process(decoding_jobs, _worker_decode, "Decoding", workers, device)
    decoding_time = time.time() - start_time
    
    # Print statistics
    success_count = sum(1 for r in results if r[2] == "success")
    skipped_count = sum(1 for r in results if r[2] == "skipped")
    error_count = sum(1 for r in results if r[2].startswith("error"))
    
    print(f"Decoding complete: {success_count} success, {skipped_count} skipped, {error_count} errors")
    print(f"Decoding time: {decoding_time:.2f}s ({len(dac_files)/decoding_time:.2f} files/sec)")
    print(f"WAV files saved to: {output_dir}")
    
    return success_count


# ---------------------------- Main Function ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Ultra-fast parallel audio encoding and decoding with DAC")
    
    # Input/Output paths
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory with audio files")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for WAV files")
    parser.add_argument("--temp_dir", required=True, type=str, default=None, help="Temporary directory for DAC files")
    
    # Performance settings
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), 
                       help=f"Number of parallel workers (default: {mp.cpu_count()})")
    parser.add_argument("--max_gpu_workers", type=int, default=8,
                       help="Maximum GPU workers (reduce if OOM errors occur)")
    
    # Model settings
    parser.add_argument("--model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    parser.add_argument("--device", type=str, default='cuda', help="'cuda' or 'cpu'. Default: auto")
    
    # Phase control
    parser.add_argument("--skip_encode", action="store_true", help="Skip encoding phase")
    parser.add_argument("--skip_decode", action="store_true", help="Skip decoding phase")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    temp_dir = pathlib.Path(args.temp_dir) if args.temp_dir else output_dir / "dac_cache"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device and adjust workers
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == 'cuda':
        effective_workers = min(args.workers, args.max_gpu_workers)
        if effective_workers < args.workers:
            print(f"Reduced workers from {args.workers} to {effective_workers} for GPU stability")
    else:
        effective_workers = args.workers
    
    print("=== Ultra-Fast DAC Processing ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}") 
    print(f"Temp: {temp_dir}")
    print(f"Device: {device}, Workers: {effective_workers}, Model: {args.model_type}")
    print(f"Encode: {not args.skip_encode}, Decode: {not args.skip_decode}")
    
    # Preload models to warm up cache
    if not args.skip_encode or not args.skip_decode:
        preload_models(args.model_type, device, effective_workers)
    
    total_processed = 0
    
    # Phase 1: Encoding
    if not args.skip_encode:
        print("\n" + "="*60)
        dac_files = encode_phase(input_dir, temp_dir, args.model_type, device, effective_workers)
        total_processed = len(dac_files)
    
    # Phase 2: Decoding  
    if not args.skip_decode:
        print("\n" + "="*60)
        decoded_count = decode_phase(temp_dir, output_dir, args.model_type, device, effective_workers)
        total_processed = decoded_count
    
    # Final summary
    print("\n" + "="*60)
    print("=== PROCESSING COMPLETE ===")
    print(f"Total files processed: {total_processed}")
    print(f"Temporary DAC files: {temp_dir}")
    print(f"Final WAV files: {output_dir}")


if __name__ == "__main__":
    # Set multiprocessing for optimal performance
    mp.set_start_method("spawn", force=True)
    
    # Increase file descriptor limit for large datasets
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard), hard))
    except (ImportError, ValueError):
        pass  # Not available on Windows
    
    main()