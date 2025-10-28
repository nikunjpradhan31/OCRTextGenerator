# OCRTextGenerator

A small tool to generate synthetic text images for OCR training from a plain-text corpus and a set of fonts. You can run the provided command-line tool ([generate_text.py](generate_text.py)) or the interactive notebook ([generate_text.ipynb](generate_text.ipynb)).

## What this repo does
- Generates images with rendered text lines sampled from a corpus (default: [corpus/ne.txt](corpus/ne.txt))
- Applies simple augmentations (skew, blur, rotation)
- Produces a ground-truth file `gt.txt` that maps image file names to their labels
- Optionally splits the dataset into train/val/test folders and creates `labels.csv` per split

## Pipeline
1. Read tokens/lines from corpus file (default: [`corpus/ne.txt`](corpus/ne.txt))
2. Choose random fonts from fonts directory (default: [`fonts/static`](fonts/static))
3. Render text with [`generate_text.py`](generate_text.py) functions:
    - `apply_skew`
    - `apply_blur`
    - `apply_rotation`
    - `center_and_resize_image`
4. Save images and write `gt.txt`
5. Optionally create dataset splits and labels CSV

## Requirements
- Python 3.8+
- Pillow, numpy
- Optional packages in notebook

## Setup

### Preparing Data
- **Corpus**: Place text file with one token/line per line (default: `corpus/ne.txt`)
- **Fonts**: Add TTF/OTF fonts to `fonts/static/` or custom directory

### Usage

#### Command Line
```bash
# Basic usage
python generate_text.py --num_images 10 --num_words 5 --image_size 256x64

# Advanced usage
python generate_text.py \
     --num_images 50000 \
     --max_lines_per_image 20 \
     --image_size 512x128 \
     --font_size_range 24-40 \
     --corpus_text my_text.txt \
     --fonts_dir my_fonts/ \
     --dataset_dir my_dataset/
```

Key parameters:
- `--num_images`: Total images to generate
- `--image_size`: Width x Height (e.g., 256x64)
- `--font_size_range`: Min-Max size (e.g., 32-48)
- `--corpus_text`: Input text file path
- `--fonts_dir`: Fonts directory path

#### Jupyter Notebook
Use [generate_text.ipynb](generate_text.ipynb) for interactive workflow.

## Output Structure
```
data/
├── images/
│ ├── image_00000.jpg
│ ├── image_00001.jpg
│ └── ...
└── gt.txt
```
```
dataset/
├── train/
│   ├── images/
│   ├── gt.txt
│   └── labels.csv
├── val/
│   └── [same structure]
└── test/
     └── [same structure]
```
