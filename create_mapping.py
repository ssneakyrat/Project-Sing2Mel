import yaml
import glob
import os

def scan_data(dataset_dir):
    # Verify dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Find all singer directories
    singer_dirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(singer_dirs)} singers: {[os.path.basename(d) for d in singer_dirs]}")
    
    if not singer_dirs:
        raise ValueError(f"No singer directories found in {dataset_dir}")
            
    # Create singer ID mapping
    singer_map = {os.path.basename(s): i for i, s in enumerate(sorted(singer_dirs))}
    
    # Find all language directories and create language mapping
    language_dirs = []
    for singer_dir in singer_dirs:
        lang_dirs = [d for d in glob.glob(os.path.join(singer_dir, "*")) if os.path.isdir(d)]
        language_dirs.extend(lang_dirs)
    
    lang_map ={os.path.basename(s): i for i, s in enumerate(sorted(language_dirs))}
    
    print(f"Found {len(language_dirs)} language: {[os.path.basename(d) for d in language_dirs]}")

    unique_languages = set(os.path.basename(l) for l in language_dirs)
    language_map = {lang: i for i, lang in enumerate(sorted(unique_languages))}
    
    # Scan for all phonemes
    all_phones = set()
    
    for singer_dir in singer_dirs:
        singer_id = os.path.basename(singer_dir)
        
        for lang_dir in glob.glob(os.path.join(singer_dir, "*")):
            if not os.path.isdir(lang_dir):
                continue
                
            language_id = os.path.basename(lang_dir)
            
            # Check for lab directory
            lab_dir = os.path.join(lang_dir, "lab")
            if not os.path.exists(lab_dir):
                continue
            
            # List lab files
            lab_files = glob.glob(os.path.join(lab_dir, "*.lab"))
            
            # Read phones from files
            for lab_file in lab_files:
                try:
                    with open(lab_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 3:
                                _, _, phone = parts
                                all_phones.add(phone)
                except Exception as e:
                    print(f"Error reading lab file {lab_file}: {str(e)}")
    
    # Create phone mapping
    phone_map = {phone: i+1 for i, phone in enumerate(sorted(all_phones))}
    print(f"Found {len(phone_map)} unique phones")
    
    data = {
        'singer_map': singer_map,
        'lang_map': lang_map,
        'phone_map': phone_map
    }

    with open('mappings.yaml', 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    return {
        'singer_map': singer_map,
        'language_map': language_map,
        'phone_map': phone_map
    }

def main():
    dataset_dir = 'datasets/'
    scan_data(dataset_dir)

if __name__ == "__main__":
    main()