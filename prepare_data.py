import os
import json
import argparse
# import urllib.request
# import subprocess
import sys
from data.Extract_raw_data_from_pkl import extract_raw_data
from data.Generate_train_val_test import process_data_split
import numpy as np


class DataPreparator:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_default_config(self):
        return {
            "data_download": {
                "raw_data_url": "https://uchicago.box.com/s/pzrig7bi14ktjy9sl87si43jg3k7ukil",
                "test_x_url": "https://uchicago.box.com/s/hvihloa9uc02ej9j5hfs1r5w07as6zj5",
                "test_y_url": "https://uchicago.box.com/s/noa156518fla2g05q1b8bnkw5blgbr1m",
                "raw_data_filename": "sample_lfp_data_20240513.pkl",
                "test_x_filename": "x_test_raw_nz04_6111595_20221222_8_theta.npy",
                "test_y_filename": "y_test_filtered_nz04_6111595_20221222_8_theta.npy"
            },
            "extraction": {
                "input_file": "sample_lfp_data_20240513.pkl",
                "output_file": "Rodents_Raw_Data.npy",
                "rodent_index": 0
            },
            "processing": {
                "mode": "Train_Val_Test",
                "output_dir": "./data",
                "sampling_rate": 1500,
                "window_size": 512,
                "step_size": 1,
                "lowcut": 6,
                "highcut": 9,
                "num_splits": 6,
                "split_all_factor": 1.0,
                "split_train_factor": 0.8
            }
        }
    
    def download_file(self, url, filename):
        """Download a file from URL to the data directory."""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists. Skipping download.")
            return filepath
        
        print(f"Downloading {filename} from {url}...")
        
        # Users will need to download manually or provide direct links
        print(f"Please manually download {filename} from:")
        print(f"  {url}")
        print(f"  Save it to: {filepath}")
        print("Press Enter when download is complete...")
        input()
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} not found after download. Please check the file path.")
        
        return filepath
    
    def download_data(self):
        """Download all required data files."""
        print("=== Downloading Data ===")
        
        download_config = self.config['data_download']
        
        # Download training data
        print("Downloading training data...")
        raw_data_path = self.download_file(
            download_config['raw_data_url'], 
            download_config['raw_data_filename']
        )
        
        # Download test data
        print("Downloading test data...")
        test_x_path = self.download_file(
            download_config['test_x_url'], 
            download_config['test_x_filename']
        )
        test_y_path = self.download_file(
            download_config['test_y_url'], 
            download_config['test_y_filename']
        )
        
        return raw_data_path, test_x_path, test_y_path
    
    def extract_data(self, raw_data_path):
        """Extract raw data from pickle file."""
        print("=== Extracting Data ===")
        
        extraction_config = self.config['extraction']
        output_path = os.path.join(self.data_dir, extraction_config['output_file'])
        
        if os.path.exists(output_path):
            print(f"Extracted data already exists at {output_path}. Skipping extraction.")
            return output_path
        
        extract_raw_data(
            input_file=raw_data_path,
            output_file=output_path,
            rodent_index=extraction_config['rodent_index']
        )
        
        return output_path
    
    def process_data(self, raw_data_path):
        """Process raw data into train/val/test sets."""
        print("=== Processing Data ===")
        
        processing_config = self.config['processing']
        
        # Check if processed data already exists
        output_dir = processing_config['output_dir']
        expected_files = []
        
        if processing_config['mode'] in ['Train_Val', 'Train_Val_Test']:
            expected_files.extend(['x_train.npy', 'y_train.npy', 'x_val.npy', 'y_val.npy'])
        
        if processing_config['mode'] in ['Test', 'Train_Val_Test']:
            expected_files.extend(['x_test.npy', 'y_test.npy'])
        
        all_exist = all(os.path.exists(os.path.join(output_dir, f)) for f in expected_files)
        
        if all_exist:
            print(f"Processed data already exists in {output_dir}. Skipping processing.")
            return
        
        # Load raw data
        raw_signal = np.load(raw_data_path)
        print(f"Loaded raw signal with shape: {raw_signal.shape}")
        
        # Process data
        process_data_split(
            raw_signal=raw_signal,
            mode=processing_config['mode'],
            sampling_rate=processing_config['sampling_rate'],
            window_size=processing_config['window_size'],
            step_size=processing_config['step_size'],
            lowcut=processing_config['lowcut'],
            highcut=processing_config['highcut'],
            num_splits=processing_config['num_splits'],
            split_all_factor=processing_config['split_all_factor'],
            split_train_factor=processing_config['split_train_factor'],
            output_dir=output_dir
        )
    
    def verify_data(self):
        """Verify that all required data files exist."""
        print("=== Verifying Data ===")
        
        processing_config = self.config['processing']
        output_dir = processing_config['output_dir']
        
        required_files = []
        
        if processing_config['mode'] in ['Train_Val', 'Train_Val_Test']:
            required_files.extend(['x_train.npy', 'y_train.npy', 'x_val.npy', 'y_val.npy'])
        
        if processing_config['mode'] in ['Test', 'Train_Val_Test']:
            required_files.extend(['x_test.npy', 'y_test.npy'])
        
        # Add test data files
        download_config = self.config['data_download']
        required_files.extend([
            download_config['test_x_filename'],
            download_config['test_y_filename']
        ])
        
        missing_files = []
        for filename in required_files:
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
            else:
                # Load and check shape
                try:
                    data = np.load(filepath, mmap_mode='r')
                    print(f"{filename}: {data.shape}")
                except Exception as e:
                    print(f"{filename}: Error loading - {e}")
                    missing_files.append(filename)
        
        if missing_files:
            print(f"\nMissing files: {missing_files}")
            return False
        
        print("\nAll data files verified successfully!")
        return True
    
    def run(self):
        """Run the complete data preparation pipeline."""
        print("Starting data preparation...")
        
        try:
            # Step 1: Download data
            raw_data_path, test_x_path, test_y_path = self.download_data()
            
            # Step 2: Extract data
            extracted_data_path = self.extract_data(raw_data_path)
            
            # Step 3: Process data
            self.process_data(extracted_data_path)
            
            # # Step 4: Verify data
            # if self.verify_data():
            #     print("\n Data preparation completed successfully!")
            #     print("\nYou can now run training with:")
            #     print("  python Train.py --config config/train_config.json")
            #     print("\nOr run testing with:")
            #     print("  python Test.py --config config/test_config.json")
            # else:
            #     print("\n Data preparation completed with errors. Please check the output above.")
                
        except Exception as e:
            print(f"\n Data preparation failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Prepare data for DCTNet training and testing.')
    parser.add_argument('--config', type=str, default='config/data_prep_config.json',
                       help='Path to data preparation configuration file')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download step (useful if data is already downloaded)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip extraction step (useful if data is already extracted)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip processing step (useful if data is already processed)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing data files')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        preparator = DataPreparator()
    else:
        preparator = DataPreparator(config_path=args.config)
    
    if args.verify_only:
        preparator.verify_data()
        return
    
    # Run data preparation with optional skips
    print("Starting data preparation...")
    
    try:
        if not args.skip_download:
            raw_data_path, test_x_path, test_y_path = preparator.download_data()
        else:
            print("Skipping download step...")
            # Assume files are in expected locations
            download_config = preparator.config['data_download']
            raw_data_path = os.path.join(preparator.data_dir, download_config['raw_data_filename'])
        
        if not args.skip_extraction:
            extracted_data_path = preparator.extract_data(raw_data_path)
        else:
            print("Skipping extraction step...")
            extraction_config = preparator.config['extraction']
            extracted_data_path = os.path.join(preparator.data_dir, extraction_config['output_file'])
        
        if not args.skip_processing:
            preparator.process_data(extracted_data_path)
        else:
            print("Skipping processing step...")
        
        # # Always verify at the end
        # if preparator.verify_data():
        #     print("\n Data preparation completed successfully!")
        #     print("\nNext steps:")
        #     print("1. Train the model:")
        #     print("   python Train.py --config config/train_config.json")
        #     print("2. Test the model:")
        #     print("   python Test.py --config config/test_config.json")
        # else:
        #     print("\n Data preparation completed with errors. Please check the output above.")
            
    except Exception as e:
        print(f"\n Data preparation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()