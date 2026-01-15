import sys
import os
from PIL import Image

def convert_pattern_to_png(pattern_file, output_dir):
    """Convert a single pattern file to PNG in the output directory"""
    # Read pattern from file
    with open(pattern_file, 'r') as f:
        lines = f.readlines()

    # Create 9x9 image
    img = Image.new('RGB', (9, 9))
    pixels = img.load()

    # Process each line and set pixel colors
    for y, line in enumerate(lines):
        line = line.strip()
        for x, char in enumerate(line):
            if char == '0':
                pixels[x, y] = (255, 255, 255)  # White
            elif char == '1':
                pixels[x, y] = (0, 0, 0)  # Black

    # Generate output filename based on input filename
    base_name = os.path.splitext(os.path.basename(pattern_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.png")

    # Save the image
    img.save(output_file)
    return output_file

def batch_convert(source_dir, target_dir):
    """Convert all pattern files from source directory to PNG in target directory"""
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")
    
    # Get all files from source directory
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        sys.exit(1)
    
    # Find all text pattern files
    pattern_files = [f for f in os.listdir(source_dir) 
                     if f.endswith('.txt') or f.endswith('.pat')]
    
    if not pattern_files:
        print(f"No pattern files found in '{source_dir}'")
        sys.exit(1)
    
    print(f"Found {len(pattern_files)} pattern file(s) in '{source_dir}'")
    print(f"Converting to PNG in '{target_dir}'...\n")
    
    # Convert each pattern file
    converted_count = 0
    for pattern_file in pattern_files:
        try:
            full_path = os.path.join(source_dir, pattern_file)
            output_file = convert_pattern_to_png(full_path, target_dir)
            print(f"✓ {pattern_file} → {os.path.basename(output_file)}")
            converted_count += 1
        except Exception as e:
            print(f"✗ Error converting {pattern_file}: {e}")
    
    print(f"\nConversion complete! {converted_count}/{len(pattern_files)} files converted.")

if __name__ == "__main__":
    # Check if both arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_directory> <target_directory>")
        print("Example: python script.py patterns/ images/")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    target_directory = sys.argv[2]
    
    batch_convert(source_directory, target_directory)
