import random
import os

def create_conifer_pattern():
    """Generate a vertically symmetrical conifer (triangle/tree) pattern"""
    pattern = []
    
    # Conifer templates - vertically symmetrical triangular shapes
    templates = [
        # Classic triangle tree
        [
            "000010000",
            "000111000",
            "001111100",
            "001111100",
            "011111110",
            "011111110",
            "111111111",
            "000010000",
            "000111000"
        ],
        # Narrow tall tree
        [
            "000010000",
            "000111000",
            "000111000",
            "001111100",
            "001111100",
            "001111100",
            "011111110",
            "000010000",
            "000010000"
        ],
        # Wide short tree
        [
            "000000000",
            "000010000",
            "000111000",
            "001111100",
            "011111110",
            "111111111",
            "111111111",
            "000010000",
            "000111000"
        ],
        # Layered tree
        [
            "000010000",
            "000111000",
            "001111100",
            "000111000",
            "001111100",
            "011111110",
            "011111110",
            "000010000",
            "000111000"
        ]
    ]
    
    # Randomly select a template with small variations
    base = random.choice(templates)
    
    # Optionally add small symmetric variations
    if random.random() > 0.5:
        pattern = base
    else:
        # Create variation by modifying symmetrically
        pattern = []
        for row in base:
            new_row = list(row)
            # Randomly remove symmetric pixels (make tree less dense)
            for i in range(4):
                if random.random() > 0.7 and new_row[i] == '1':
                    new_row[i] = '0'
                    new_row[8-i] = '0'
            pattern.append(''.join(new_row))
    
    return pattern

def create_non_conifer_pattern():
    """Generate non-conifer patterns (circles, squares, random shapes)"""
    pattern_type = random.choice(['circle', 'square', 'random', 'horizontal'])
    pattern = []
    
    if pattern_type == 'circle':
        # Approximate circle
        circle_templates = [
            [
                "000111000",
                "011111110",
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "011111110",
                "000111000"
            ],
            [
                "000010000",
                "001111100",
                "011111110",
                "011111110",
                "111111111",
                "011111110",
                "011111110",
                "001111100",
                "000010000"
            ]
        ]
        pattern = random.choice(circle_templates)
    
    elif pattern_type == 'square':
        # Square or rectangle
        square_templates = [
            [
                "000000000",
                "011111110",
                "011111110",
                "011111110",
                "011111110",
                "011111110",
                "011111110",
                "011111110",
                "000000000"
            ],
            [
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "111111111",
                "111111111"
            ]
        ]
        pattern = random.choice(square_templates)
    
    elif pattern_type == 'horizontal':
        # Horizontal lines or patterns
        pattern = [
            "000000000",
            "111111111",
            "111111111",
            "000000000",
            "000000000",
            "111111111",
            "111111111",
            "000000000",
            "000000000"
        ]
    
    else:  # random
        # Random asymmetric pattern
        pattern = []
        for _ in range(9):
            row = ''.join([str(random.randint(0, 1)) for _ in range(9)])
            pattern.append(row)
    
    return pattern

def save_pattern(pattern, filename):
    """Save pattern to file"""
    with open(filename, 'w') as f:
        for line in pattern:
            f.write(line + '\n')

def generate_dataset(num_conifers=50, num_non_conifers=50, output_dir='patterns'):
    """Generate a dataset of conifer and non-conifer patterns"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate conifer patterns
    print(f"Generating {num_conifers} conifer patterns...")
    for i in range(num_conifers):
        pattern = create_conifer_pattern()
        filename = os.path.join(output_dir, f'conifer_{i:03d}.txt')
        save_pattern(pattern, filename)
    
    # Generate non-conifer patterns
    print(f"Generating {num_non_conifers} non-conifer patterns...")
    for i in range(num_non_conifers):
        pattern = create_non_conifer_pattern()
        filename = os.path.join(output_dir, f'non_conifer_{i:03d}.txt')
        save_pattern(pattern, filename)
    
    print(f"Dataset generated in '{output_dir}' directory!")
    print(f"Total patterns: {num_conifers + num_non_conifers}")

# Generate the dataset
if __name__ == "__main__":
    generate_dataset(num_conifers=50, num_non_conifers=50)