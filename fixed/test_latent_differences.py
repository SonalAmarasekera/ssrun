#!/usr/bin/env python
"""
test_latent_differences.py

Test script to verify that latent files with the same name across
mix_clean, s1, and s2 directories have different content.

Quick: python test_latent_differences.py --latent_dir ./latents --quick --num_samples 20
Detailed: python test_latent_differences.py --latent_dir ./latents --num_samples 15
"""

import argparse
import pathlib
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_latent_file(file_path: pathlib.Path) -> Dict:
    """Load a latent .pt file and return its contents."""
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compare_latent_files(file1: pathlib.Path, file2: pathlib.Path) -> Dict:
    """
    Compare two latent files and return similarity metrics.
    """
    data1 = load_latent_file(file1)
    data2 = load_latent_file(file2)
    
    if data1 is None or data2 is None:
        return None
    
    # Extract latent tensors
    z1 = data1['z']  # [T, C]
    z2 = data2['z']  # [T, C]
    
    # Ensure same dimensions for fair comparison
    min_t = min(z1.shape[0], z2.shape[0])
    z1 = z1[:min_t]
    z2 = z2[:min_t]
    
    # Calculate similarity metrics
    metrics = {
        'file1': str(file1),
        'file2': str(file2),
        'shape1': tuple(z1.shape),
        'shape2': tuple(z2.shape),
        'mean_diff': float(torch.mean(torch.abs(z1 - z2)).item()),
        'max_diff': float(torch.max(torch.abs(z1 - z2)).item()),
        'cosine_similarity': float(torch.nn.functional.cosine_similarity(
            z1.flatten(), z2.flatten(), dim=0).item()),
        'correlation': float(torch.corrcoef(torch.stack([z1.flatten(), z2.flatten()]))[0, 1].item()),
        'identical': torch.allclose(z1, z2, rtol=1e-5, atol=1e-5),
    }
    
    return metrics


def find_matching_files(latent_dir: pathlib.Path) -> List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path]]:
    """
    Find triplets of files with the same name across mix_clean, s1, s2 directories.
    """
    mix_dir = latent_dir / "mix_clean"
    s1_dir = latent_dir / "s1" 
    s2_dir = latent_dir / "s2"
    
    if not all(d.exists() for d in [mix_dir, s1_dir, s2_dir]):
        print(f"Error: Required directories not found in {latent_dir}")
        return []
    
    # Get all .pt files in each directory
    mix_files = {f.stem: f for f in mix_dir.rglob("*.pt")}
    s1_files = {f.stem: f for f in s1_dir.rglob("*.pt")}
    s2_files = {f.stem: f for f in s2_dir.rglob("*.pt")}
    
    # Find common stems
    common_stems = set(mix_files.keys()) & set(s1_files.keys()) & set(s2_files.keys())
    
    triplets = []
    for stem in common_stems:
        triplets.append((mix_files[stem], s1_files[stem], s2_files[stem]))
    
    print(f"Found {len(triplets)} matching file triplets")
    return triplets


def analyze_triplet(mix_file: pathlib.Path, s1_file: pathlib.Path, s2_file: pathlib.Path, 
                   sample_name: str = "") -> Dict:
    """
    Analyze a triplet of files (mix, s1, s2) and return comprehensive comparison results.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {sample_name}")
    print(f"Mix: {mix_file.name}")
    print(f"S1: {s1_file.name}")
    print(f"S2: {s2_file.name}")
    
    # Compare all pairs
    comparisons = {
        'mix_vs_s1': compare_latent_files(mix_file, s1_file),
        'mix_vs_s2': compare_latent_files(mix_file, s2_file),
        's1_vs_s2': compare_latent_files(s1_file, s2_file),
    }
    
    # Load file metadata for additional verification
    mix_data = load_latent_file(mix_file)
    s1_data = load_latent_file(s1_file)
    s2_data = load_latent_file(s2_file)
    
    if all(data is not None for data in [mix_data, s1_data, s2_data]):
        print(f"\nMetadata:")
        print(f"  Mix: T={mix_data['T']}, C={mix_data['C']}, fps={mix_data['fps']}")
        print(f"  S1:  T={s1_data['T']}, C={s1_data['C']}, fps={s1_data['fps']}")
        print(f"  S2:  T={s2_data['T']}, C={s2_data['C']}, fps={s2_data['fps']}")
    
    # Print comparison results
    for comp_name, metrics in comparisons.items():
        if metrics:
            print(f"\n{comp_name}:")
            print(f"  Mean Difference: {metrics['mean_diff']:.6f}")
            print(f"  Max Difference:  {metrics['max_diff']:.6f}")
            print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
            print(f"  Correlation: {metrics['correlation']:.6f}")
            print(f"  Identical: {metrics['identical']}")
            
            # Warning if files are too similar
            if metrics['cosine_similarity'] > 0.95:
                print(f"  ⚠️  WARNING: Files are very similar!")
            if metrics['identical']:
                print(f"  ❌ ERROR: Files are identical!")
    
    return comparisons


def run_comprehensive_test(latent_dir: pathlib.Path, num_samples: int = 10):
    """
    Run comprehensive test on multiple file triplets.
    """
    print("=" * 80)
    print("LATENT FILE DIFFERENCE TEST")
    print("=" * 80)
    
    triplets = find_matching_files(latent_dir)
    
    if not triplets:
        print("No matching file triplets found!")
        return
    
    print(f"Testing {min(num_samples, len(triplets))} samples out of {len(triplets)} total triplets...")
    
    all_results = []
    test_samples = triplets[:num_samples]
    
    for i, (mix_file, s1_file, s2_file) in enumerate(test_samples):
        results = analyze_triplet(mix_file, s1_file, s2_file, f"Sample {i+1}")
        all_results.append(results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for comp_type in ['mix_vs_s1', 'mix_vs_s2', 's1_vs_s2']:
        similarities = []
        identical_count = 0
        
        for result in all_results:
            if result[comp_type]:
                similarities.append(result[comp_type]['cosine_similarity'])
                if result[comp_type]['identical']:
                    identical_count += 1
        
        if similarities:
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            min_sim = np.min(similarities)
            
            print(f"\n{comp_type}:")
            print(f"  Cosine Similarity - Avg: {avg_sim:.4f}, Min: {min_sim:.4f}, Max: {max_sim:.4f}")
            print(f"  Identical files: {identical_count}/{len(all_results)}")
            
            if identical_count > 0:
                print(f"  ❌ CRITICAL: {identical_count} file pairs are identical!")
            elif avg_sim > 0.8:
                print(f"  ⚠️  WARNING: Average similarity is high ({avg_sim:.4f})")
            else:
                print(f"  ✅ GOOD: Files appear sufficiently different")
    
    # Create visualization
    create_similarity_plot(all_results, latent_dir)


def create_similarity_plot(all_results: List[Dict], latent_dir: pathlib.Path):
    """
    Create a box plot showing similarity distributions - FIXED VERSION.
    """
    similarity_data = []
    
    for comp_type in ['mix_vs_s1', 'mix_vs_s2', 's1_vs_s2']:
        for i, result in enumerate(all_results):
            if result[comp_type]:
                sim = result[comp_type]['cosine_similarity']
                similarity_data.append({
                    'comparison': comp_type,
                    'similarity': sim,
                    'sample': i+1
                })
    
    if similarity_data:
        # Convert to DataFrame for seaborn
        df = pd.DataFrame(similarity_data)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='comparison', y='similarity')
        plt.title('Cosine Similarity Distributions Between Latent Files')
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='High Similarity Threshold')
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Medium Similarity Threshold')
        plt.legend()
        plt.tight_layout()
        
        plot_path = latent_dir / "latent_similarity_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Similarity plot saved to: {plot_path}")
        plt.show()


def quick_verification(latent_dir: pathlib.Path, num_check: int = 5):
    """
    Quick verification - check if any files are identical.
    """
    print("\n" + "="*80)
    print("QUICK IDENTICAL FILE CHECK")
    print("="*80)
    
    triplets = find_matching_files(latent_dir)
    
    if not triplets:
        return
    
    identical_count = 0
    checked_count = 0
    
    for mix_file, s1_file, s2_file in triplets[:num_check]:
        checked_count += 1
        
        # Quick load and compare
        mix_data = load_latent_file(mix_file)
        s1_data = load_latent_file(s1_file)
        s2_data = load_latent_file(s2_file)
        
        if all(data is not None for data in [mix_data, s1_data, s2_data]):
            mix_z = mix_data['z']
            s1_z = s1_data['z']
            s2_z = s2_data['z']
            
            # Check if any pair is identical
            if torch.allclose(mix_z, s1_z, rtol=1e-5, atol=1e-5):
                print(f"❌ IDENTICAL: {mix_file.name} (mix) == {s1_file.name} (s1)")
                identical_count += 1
            if torch.allclose(mix_z, s2_z, rtol=1e-5, atol=1e-5):
                print(f"❌ IDENTICAL: {mix_file.name} (mix) == {s2_file.name} (s2)")
                identical_count += 1
            if torch.allclose(s1_z, s2_z, rtol=1e-5, atol=1e-5):
                print(f"❌ IDENTICAL: {s1_file.name} (s1) == {s2_file.name} (s2)")
                identical_count += 1
    
    if identical_count == 0:
        print(f"✅ No identical files found in {checked_count} checked samples")
    else:
        print(f"❌ Found {identical_count} pairs of identical files!")


def main():
    parser = argparse.ArgumentParser(description="Test latent file differences")
    parser.add_argument("--latent_dir", required=True, help="Directory containing mix_clean, s1, s2 subdirectories")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--quick", action="store_true", help="Run quick verification only")
    
    args = parser.parse_args()
    
    latent_dir = pathlib.Path(args.latent_dir)
    
    if not latent_dir.exists():
        print(f"Error: Latent directory {latent_dir} does not exist")
        return
    
    if args.quick:
        quick_verification(latent_dir, args.num_samples)
    else:
        run_comprehensive_test(latent_dir, args.num_samples)


if __name__ == "__main__":

    main()
