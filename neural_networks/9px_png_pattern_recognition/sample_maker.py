import sys
from PIL import Image

# Check if pattern file is provided as command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <pattern_file>")
    sys.exit(1)

pattern_file = sys.argv[1]

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
import os
base_name = os.path.splitext(os.path.basename(pattern_file))[0]
output_file = f"{base_name}.png"

# Save the image
img.save(output_file)
print(f"Image saved as {output_file}")
