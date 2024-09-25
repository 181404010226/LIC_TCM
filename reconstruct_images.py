import os
import re
from PIL import Image
from collections import defaultdict

def parse_filename(filename):
    """
    Extracts the base name and tile information from the filename.
    
    Example:
    Input: Tile_+000_-029_L17_00_texture_0#0#0#64#64_decompressed.jpg
    Output: base_name = 'Tile_+000_-029_L17_00_texture_0', x=0, y=0, width=64, height=64
    """
    pattern = r"(.+?)#(\d+)#(\d+)#(\d+)#(\d+)_decompressed\.jpg"
    match = re.match(pattern, filename)
    if match:
        base_name = match.group(1)
        x = int(match.group(2))
        y = int(match.group(3))
        width = int(match.group(4))
        height = int(match.group(5))
        return base_name, x, y, width, height
    else:
        return None

def traverse_images_directory(images_dir):
    """
    Traverses the images directory and groups tiles by their base image name.
    Also keeps track of the relative paths to preserve folder structure.
    
    Returns a dictionary with base_name as keys and list of tile info as values.
    Each tile info includes the path, x, y, width, height, and relative directory.
    """
    tiles_dict = defaultdict(list)
    base_name_to_relpath = {}

    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith("_decompressed.jpg"):
                parsed = parse_filename(file)
                if parsed:
                    base_name, x, y, width, height = parsed
                    tile_path = os.path.join(root, file)
                    # Compute relative path
                    rel_dir = os.path.relpath(root, images_dir)
                    tiles_dict[base_name].append({
                        'path': tile_path,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'rel_dir': rel_dir
                    })
                else:
                    print(f"Filename parsing failed for: {file}")
    
    return tiles_dict

def reconstruct_image(base_name, tiles, images_dir, output_dir):
    """
    Reconstructs the original image from its tiles and saves it to the output directory.
    Preserves the folder structure relative to images_dir.
    """
    # Determine the original image size
    max_width = max(tile['width'] for tile in tiles)
    max_height = max(tile['height'] for tile in tiles)
    original_size = (max_width, max_height)
    reconstructed = Image.new('RGB', original_size)
    
    for tile in tiles:
        try:
            with Image.open(tile['path']) as img:
                # Paste the tile at the specified position
                reconstructed.paste(img, (tile['x'], tile['y']))
        except Exception as e:
            print(f"Error processing {tile['path']}: {e}")
    
    # Preserve the relative directory structure
    rel_dir = tiles[0]['rel_dir']  # All tiles have the same relative directory
    output_subdir = os.path.join(output_dir, rel_dir)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, f"{base_name}.jpg")
    
    try:
        reconstructed.save(output_path)
        print(f"Saved restored image: {output_path}")
    except Exception as e:
        print(f"Failed to save image {output_path}: {e}")

def main():
    images_dir = "3dtilesCompressedBIN/images"
    output_dir = "restored_images"
    
    tiles_dict = traverse_images_directory(images_dir)
    
    for base_name, tiles in tiles_dict.items():
        reconstruct_image(base_name, tiles, images_dir, output_dir)

if __name__ == "__main__":
    main()