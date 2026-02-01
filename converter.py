import cv2
import numpy as np
import argparse
import os
import sys

def convert_mp4_to_npy(input_path, output_path, grayscale=False):
    """
    Reads an MP4 video, flattens each frame, and saves as a 2D numpy array.
    
    Args:
        input_path (str): Path to the input MP4 file.
        output_path (str): Path to save the output .npy file.
        resize (tuple): Optional (width, height) to resize frames before processing.
        grayscale (bool): Whether to convert frames to grayscale.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: Input file '{input_path}' not found.")

    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file '{input_path}'.")

    frames_data = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing '{input_path}'...")
    print(f"Original Resolution: {width}x{height}")
    print(f"Total Frames: {frame_count}")
    

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Flatten the frame
            # If color: (H, W, 3) -> (H*W*3,)
            # If grayscale: (H, W) -> (H*W,)
            flat_frame = frame.flatten()
            frames_data.append(flat_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames...", end='\r')

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        cap.release()

    print(f"\nFinished processing {frame_idx} frames.")
    
    if not frames_data:
        raise ValueError("No frames extracted from video.")

    # Stack into 2D array
    result_array = np.vstack(frames_data)
    
    # Save to file
    np.save(output_path, result_array)
    print(f"Saved output to '{output_path}'")
    print(f"Output shape: {result_array.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 video frames to a flattened 2D numpy array.")
    parser.add_argument("input_file", help="Path to the input MP4 file")
    parser.add_argument("output_file", help="Path to the output .npy file")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale before flattening")

    args = parser.parse_args()

    convert_mp4_to_npy(args.input_file, args.output_file, grayscale=args.grayscale)

def convert_sequence_to_npy(input_folder, output_path, grayscale=False):
    """
    Reads a sequence of images from a folder, flattens each, and saves as a 2D numpy array.
    
    Args:
        input_folder (str): Path to the folder containing images.
        output_path (str): Path to save the output .npy file.
        grayscale (bool): Whether to convert frames to grayscale.
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Error: Input folder '{input_folder}' not found.")

    import re

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    # Get list of image files
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tga')
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)], key=natural_sort_key)
    
    if not files:
        raise ValueError(f"Error: No image files found in '{input_folder}'.")

    frames_data = []
    width = 0
    height = 0
    
    print(f"Processing folder '{input_folder}'...")
    print(f"Found {len(files)} images.")

    for i, fname in enumerate(files):
        fpath = os.path.join(input_folder, fname)
        frame = cv2.imread(fpath)
        
        if frame is None:
            print(f"Warning: Could not read image '{fname}', skipping.")
            continue
            
        if width == 0:
            height, width = frame.shape[:2]
            print(f"Resolution: {width}x{height}")
            
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
            
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        flat_frame = frame.flatten()
        frames_data.append(flat_frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} frames...", end='\r')

    print(f"\nFinished processing {len(frames_data)} frames.")
    
    if not frames_data:
        raise ValueError("No frames extracted from sequence.")

    result_array = np.vstack(frames_data)
    np.save(output_path, result_array)
    print(f"Saved output to '{output_path}'")
    print(f"Output shape: {result_array.shape}")
