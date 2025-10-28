import os
import unicodedata
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import csv
import argparse
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate text images for OCR training')
    
    # Required arguments
    parser.add_argument('--num_images', type=int, required=True, help='Total number of images to generate')
    parser.add_argument('--num_words', type=int, required=True, help='Number of words per image')
    parser.add_argument('--image_size', type=str, required=True, help='Image size as WxH (e.g., 256x64)')
    parser.add_argument('--font_size_range', type=str, required=True, help='Font size range as min-max (e.g., 32-48)')
    
    # Optional arguments with defaults
    parser.add_argument('--corpus_text', type=str, default='corpus/corpus.txt', help='Path to text corpus file')
    parser.add_argument('--fonts_dir', type=str, default='fonts/static', help='Directory containing font files')
    parser.add_argument('--output_dir', type=str, default='data/images', help='Output directory for images')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Dataset directory for metadata')
    parser.add_argument('--output_text', type=str, default='data', help='Output directory for text files')

    # Augmentation randomness parameters (0-1)
    parser.add_argument('--skew_prob', type=float, default=0.5, help='Probability of applying skew (0-1)')
    parser.add_argument('--blur_prob', type=float, default=0.5, help='Probability of applying blur (0-1)')
    parser.add_argument('--rotation_prob', type=float, default=0.3, help='Probability of applying rotation (0-1)')
    parser.add_argument('--distortion_prob', type=float, default=0.0, help='Probability of applying distortion (0-1)')
    
    # Augmentation intensity parameters
    parser.add_argument('--max_skew_angle', type=float, default=5.0, help='Maximum skew angle in degrees')
    parser.add_argument('--max_blur_radius', type=float, default=1.5, help='Maximum blur radius')
    parser.add_argument('--max_rotation_angle', type=float, default=5.0, help='Maximum rotation angle in degrees')
    

    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio (0-1)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio (0-1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio (0-1)')
    return parser.parse_args()

def apply_skew(img, max_angle=5):
    angle = random.uniform(-max_angle, max_angle)
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, math.tan(math.radians(angle)), 0, 0, 1, 0),
        resample=Image.BICUBIC
    )

def apply_blur(img, max_radius=1.5):
    radius = random.uniform(0, max_radius)
    if radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img

def apply_rotation(img, max_angle=5):
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, expand=True, fillcolor="white")

def decenter_text(x, y, text_width, text_height, img_width, img_height, margin=10):
    max_x = max(img_width - text_width - margin, margin)
    max_y = max(img_height - text_height - margin, margin)
    x = random.randint(margin, max_x)
    y = random.randint(margin, max_y)
    return x, y

def center_and_resize_image(img, target_size=(1156, 64)):
    target_w, target_h = target_size
    img_w, img_h = img.size

    if img_w > target_w or img_h > target_h:
        img.thumbnail((target_w, target_h), Image.LANCZOS)

    new_img = Image.new("RGB", (target_w, target_h), color="black")

    paste_x = (target_w - img.width) // 2
    paste_y = (target_h - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))

    return new_img

def draw_text(text, fontfile, output_path, base_image_size, args):
    text = unicodedata.normalize("NFC", text)
    
    font_size_min, font_size_max = map(int, args.font_size_range.split('-'))
    
    font_size = random.randint(font_size_min, font_size_max)
    pil_font = ImageFont.truetype(fontfile, font_size)
    
    temp_img = Image.new("RGB", base_image_size, color="white")
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=pil_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    min_font_size = max(20, font_size_min)
    while text_width + 20 > base_image_size[0] and font_size > min_font_size:
        font_size -= 1
        pil_font = ImageFont.truetype(fontfile, font_size)
        bbox = draw.textbbox((0, 0), text, font=pil_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    
    img_width = max(base_image_size[0], text_width + 20)
    img_height = base_image_size[1]
    
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)
    
    x, y = decenter_text(0, 0, text_width, text_height, img_width, img_height)
    
    draw.text((x, y), text, font=pil_font, fill="black")
    if random.random() < args.skew_prob:
        img = apply_skew(img, args.max_skew_angle)
    if random.random() < args.blur_prob:
        img = apply_blur(img, args.max_blur_radius)
    if random.random() < args.rotation_prob:
        img = apply_rotation(img, args.max_rotation_angle)
    
    img = center_and_resize_image(img, (1156, 64))
    img.save(output_path)

def generate_labels_csv(base_dir):
    """
    For each split folder (train, val, test) inside base_dir,
    reads gt.txt and creates labels.csv with columns: filename, words
    """
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        gt_path = os.path.join(split_dir, "gt.txt")
        csv_path = os.path.join(split_dir, "labels.csv")

        if not os.path.exists(gt_path):
            print(f"⚠️ No gt.txt found for {split}")
            continue

        rows = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                filename, text = line.split(" ", 1)
                rows.append((filename, text))

        # Write CSV
        with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "words"])
            writer.writerows(rows)

        print(f"✅ Created {csv_path} with {len(rows)} entries")


def create_dataset_splits(args):
    """Create train/val/test splits and organize files according to notebook structure"""
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.dataset_dir, split), exist_ok=True)
    
    main_gt_path = os.path.join(args.output_text, "gt.txt")
    if not os.path.exists(main_gt_path):
        print(f"Error: Main gt.txt file not found at {main_gt_path}")
        return
    
    entries = []
    with open(main_gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                filename, text = line.split(" ", 1)
                entries.append((filename, text))
    
    random.shuffle(entries)
    
    total = len(entries)
    train_size = int(total * args.train_ratio)
    val_size = int(total * args.val_ratio)
    test_size = total - train_size - val_size
    
    train_entries = entries[:train_size]
    val_entries = entries[train_size:train_size + val_size]
    test_entries = entries[train_size + val_size:]
    
    def process_split(split_entries, split_name):
        split_dir = os.path.join(args.dataset_dir, split_name)
        gt_path = os.path.join(split_dir, "gt.txt")
        
        with open(gt_path, "w", encoding="utf-8") as f:
            for filename, text in split_entries:
                f.write(f"{filename} {text}\n")
                
                src_path = os.path.join(args.output_dir, filename)
                dst_path = os.path.join(split_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
        
        print(f"✅ Created {split_name} set with {len(split_entries)} images")
    
    process_split(train_entries, "train")
    process_split(val_entries, "val")
    process_split(test_entries, "test")
    
    generate_labels_csv(args.dataset_dir)


def main():
    args = parse_arguments()
    
    image_width, image_height = map(int, args.image_size.split('x'))
    image_size = (image_width, image_height)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_text, exist_ok=True)
    
    try:
        with open(args.corpus_text, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Corpus file '{args.corpus_text}' not found.")
        return
    
    font_files = []
    if os.path.exists(args.fonts_dir):
        font_files = [os.path.join(args.fonts_dir, f) for f in os.listdir(args.fonts_dir)
                      if f.lower().endswith((".ttf", ".otf"))]
    
    if not font_files:
        print(f"Error: No font files found in '{args.fonts_dir}'")
        return
    
    gt_file_path = os.path.join(args.output_text, "gt.txt")
    
    counter = 0
    with open(gt_file_path, "w", encoding="utf-8") as gt_file:
        for _ in range(args.num_images):
            num_lines = random.randint(1, args.num_words)
            selected_lines = random.choices(lines, k=num_lines)
            text = " ".join(selected_lines) 
            
            font_file = random.choice(font_files)
            file_name = f"image_{counter:05d}.jpg"
            output_path = os.path.join(args.output_dir, file_name)
            
            draw_text(text, font_file, output_path, image_size, args)
            
            gt_file.write(f"{file_name} {text}\n")
            
            counter += 1
            if counter % 1000 == 0:
                print(f"Generated {counter}/{args.num_images} images...")
    
    print(f"✅ Generated {counter} images in {args.output_dir}")
    print(f"✅ GT file saved at: {gt_file_path}")
    
    print("Applying final center and resize to all images...")
    for file in os.listdir(args.output_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(args.output_dir, file)
            img = Image.open(img_path)
            img = center_and_resize_image(img, (1156, 64))
            img.save(img_path)
    
    print("Creating dataset splits...")
    create_dataset_splits(args)
    
    print("✅ Dataset generation complete!")
    print(f"   - Original images: {args.output_dir}")
    print(f"   - Main gt.txt: {gt_file_path}")
    print(f"   - Dataset splits: {args.dataset_dir}/{{train,val,test}}/")
    print(f"   - Each split contains: gt.txt and labels.csv")

if __name__ == "__main__":
    main()