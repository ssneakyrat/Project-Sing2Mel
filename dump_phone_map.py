#!/usr/bin/env python
"""
Script to dump the phone map from the SingingVoiceDataset and separate it by language.
This will help understand what phonemes are available for the singing synthesizer.
"""
import torch
import os
import pickle
import json
from collections import defaultdict
import argparse
import yaml

from dataset import get_dataloader
from data_utils import standardized_collate_fn

def find_dataset_cache_files(cache_dir="./cache"):
    """Find all dataset cache files in the cache directory."""
    cache_files = []
    for file in os.listdir(cache_dir):
        if file.endswith(".pkl") and "singing_voice" in file:
            cache_files.append(os.path.join(cache_dir, file))
    return cache_files

def load_dataset_cache(cache_file):
    """Load a dataset cache file."""
    print(f"Loading dataset cache: {cache_file}")
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data
    except Exception as e:
        print(f"Error loading cache file {cache_file}: {e}")
        return None

def analyze_phone_map(cache_data):
    """Analyze the phone map and phone statistics by language."""
    if not cache_data:
        return None
    
    # Extract maps
    phone_map = cache_data.get('phone_map', {})
    inv_phone_map = cache_data.get('inv_phone_map', {})
    language_map = cache_data.get('language_map', {})
    inv_language_map = cache_data.get('inv_language_map', {})
    phone_language_stats = cache_data.get('phone_language_stats', {})
    
    # Create phone usage statistics by language
    phones_by_language = defaultdict(set)
    phone_counts_by_language = defaultdict(lambda: defaultdict(int))
    
    # Process phone_language_stats to get phone counts by language
    for lang_id, phone_counts in phone_language_stats.items():
        lang_name = lang_id  # Default to ID if name not found
        if lang_id in inv_language_map:
            lang_name = inv_language_map[lang_id]
        
        for phone, count in phone_counts.items():
            phones_by_language[lang_name].add(phone)
            phone_counts_by_language[lang_name][phone] = count
    
    # Convert sets to lists for JSON serialization
    result = {
        "phone_map": phone_map,
        "inv_phone_map": inv_phone_map,
        "language_map": language_map,
        "inv_language_map": inv_language_map, 
        "phones_by_language": {lang: sorted(list(phones)) for lang, phones in phones_by_language.items()},
        "phone_counts_by_language": {lang: dict(counts) for lang, counts in phone_counts_by_language.items()}
    }
    
    return result

def save_phone_maps(analysis_results, output_dir="./phone_maps"):
    """Save phone maps to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full analysis as JSON
    full_analysis_path = os.path.join(output_dir, "full_phone_analysis.json")
    with open(full_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"Full analysis saved to: {full_analysis_path}")
    
    # Save individual language phoneme sets
    for lang, phones in analysis_results["phones_by_language"].items():
        lang_file = os.path.join(output_dir, f"{lang}_phonemes.txt")
        with open(lang_file, 'w', encoding='utf-8') as f:
            # Get phone counts if available
            phone_counts = analysis_results["phone_counts_by_language"].get(lang, {})
            
            # Write header
            f.write(f"# Phoneme set for language: {lang}\n")
            f.write("# Format: phoneme | count | phoneme_id\n\n")
            
            # Write phonemes with their counts and IDs
            for phone in sorted(phones):
                count = phone_counts.get(phone, 0)
                phone_id = analysis_results["phone_map"].get(phone, "")
                f.write(f"{phone} | {count} | {phone_id}\n")
        
        print(f"Phoneme set for {lang} saved to: {lang_file}")
    
    # Save a summary file
    summary_path = os.path.join(output_dir, "phoneme_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Phoneme Summary Across Languages\n\n")
        
        # Get all unique phonemes
        all_phonemes = set()
        for phones in analysis_results["phones_by_language"].values():
            all_phonemes.update(phones)
        
        # Write summary table header
        f.write("Phoneme | Languages\n")
        f.write("--------|----------\n")
        
        # For each phoneme, list languages that use it
        for phone in sorted(all_phonemes):
            langs_using_phone = [lang for lang, phones in analysis_results["phones_by_language"].items() if phone in phones]
            f.write(f"{phone} | {', '.join(langs_using_phone)}\n")
    
    print(f"Phoneme summary saved to: {summary_path}")

def generate_ipa_mapping_template(analysis_results, output_dir="./phone_maps"):
    """
    Generate a template YAML file for mapping dataset phonemes to IPA symbols.
    This can be manually edited later.
    """
    # Get all unique phonemes across all languages
    all_phonemes = set()
    for phones in analysis_results["phones_by_language"].values():
        all_phonemes.update(phones)
    
    # Create template mapping (phoneme -> empty string to be filled)
    ipa_mapping = {phone: "" for phone in sorted(all_phonemes)}
    
    # Add some common mappings as examples
    common_mappings = {
        "a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
        "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "ɡ",
        "s": "s", "z": "z", "m": "m", "n": "n", "l": "l", "r": "r"
    }
    
    for phone, ipa in common_mappings.items():
        if phone in ipa_mapping:
            ipa_mapping[phone] = ipa
    
    # Save the template file
    template_path = os.path.join(output_dir, "ipa_mapping_template.yaml")
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write("# Template for mapping dataset phonemes to IPA symbols\n")
        f.write("# Fill in the IPA value for each phoneme\n\n")
        yaml.dump(ipa_mapping, f, default_flow_style=False, allow_unicode=True)
    
    print(f"IPA mapping template saved to: {template_path}")

def main():
    parser = argparse.ArgumentParser(description="Dump phone map from SingingVoiceDataset and separate by language")
    parser.add_argument("--cache-dir", default="./cache", help="Directory containing dataset cache files")
    parser.add_argument("--output-dir", default="./phone_maps", help="Directory to save phone maps")
    parser.add_argument("--generate-ipa", action="store_true", help="Generate IPA mapping template")
    
    args = parser.parse_args()
    
    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=16,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
        train_files=100,
        val_files=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        collate_fn=standardized_collate_fn,  # Use the standardized collate function from utils.py
        context_window_sec=5,  # Pass context window from config
        seed=42
    )

    # Find all cache files
    cache_files = find_dataset_cache_files(args.cache_dir)

    if not cache_files:
        print(f"No dataset cache files found in {args.cache_dir}")
        return
    
    # Load and analyze the first cache file found
    # We assume phone maps are the same across train/validation splits
    cache_data = load_dataset_cache(cache_files[0])
    analysis_results = analyze_phone_map(cache_data)
    
    if analysis_results:
        save_phone_maps(analysis_results, args.output_dir)
        
        if args.generate_ipa:
            generate_ipa_mapping_template(analysis_results, args.output_dir)
        
        print("\nSummary:")
        print(f"Found {len(analysis_results['phone_map'])} unique phonemes")
        print(f"Found {len(analysis_results['language_map'])} languages:")
        for lang_id, lang_name in analysis_results['inv_language_map'].items():
            num_phones = len(analysis_results['phones_by_language'].get(lang_name, []))
            print(f"  - {lang_name}: {num_phones} phonemes")
    else:
        print("Failed to analyze phone map from cache files")

if __name__ == "__main__":
    main()
