
import numpy as np
import os
import pickle
import argparse

def extract_raw_data(input_file, output_file, rodent_index=0):
    with open(input_file, 'rb') as file:
        data = pickle.load(file)

    rodent_ids = list(data.keys())
    print(f"Available rodent IDs: {rodent_ids}")

    if rodent_index >= len(rodent_ids):
        raise ValueError(f"Rodent index {rodent_index} out of range. Available indices: 0-{len(rodent_ids)-1}")

    rodent_id = rodent_ids[rodent_index]
    print(f"Processing rodent ID: {rodent_id}")

    raw_signal = data[rodent_id]['raw_signal']
    print(f"Raw signal shape: {raw_signal.shape}")

    np.save(output_file, raw_signal)
    print(f"Data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract raw data from pickle file.')
    parser.add_argument('--input', default='sample_lfp_data_20240513.pkl', 
                       help='Input pickle file path')
    parser.add_argument('--output', default='Rodents_Single_Raw_Data.npy', 
                       help='Output numpy file path')
    parser.add_argument('--rodent_index', type=int, default=0, 
                       help='Index of rodent to extract')
    
    args = parser.parse_args()
    extract_raw_data(args.input, args.output, args.rodent_index)

if __name__ == '__main__':
    main()
