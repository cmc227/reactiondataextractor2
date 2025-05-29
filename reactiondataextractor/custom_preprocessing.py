import cv2
import numpy as np
import os
import sys
import imageio
from processors import ImageReader, ImageScaler, ImageNormaliser, Binariser

def preprocess_for_arrows(image_path: str):
    print(f"[Preprocessing] Arrows: Loading and processing image {image_path}")

    filename = os.path.basename(image_path)
    if not filename.lower().endswith(('png', '.jpg', '.jpeg', '.gif')):
        print(f'Unsupported file type: {filename}')
        return None

    if filename.lower().endswith('.gif'):
        gif = imageio.mimread(image_path)
        img = np.array(gif[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f'Failed to load image: {filename}')
        return None
    
    kernel = np.array([[1, -2, 1],
                       [-2, 5, -2],
                       [1, -2, 1]])
    arrow_image = cv2.filter2D(img, -1, kernel)

    print(f"[Preprocessing] Arrows: Complete")

    return arrow_image

def preprocess_for_diagrams(image_path: str):
    print(f"[Preprocessing] Diagrams: Loading and processing image {image_path}")

    filename = os.path.basename(image_path)
    if not filename.lower().endswith(('png', '.jpg', '.jpeg', '.gif')):
        print(f'Unsupported file type: {filename}')
        return None

    if filename.lower().endswith('.gif'):
        gif = imageio.mimread(image_path)
        img = np.array(gif[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f'Failed to load image: {filename}')
        return None
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    diagram_image = cv2.filter2D(img, -1, kernel)

    print(f"[Preprocessing] Diagrams: Complete")

    return diagram_image


def preprocess_for_labels(image_path: str, sr):
    print(f"[Preprocessing] Labels: Loading and processing image {image_path}")

    filename = os.path.basename(image_path)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        print(f"[Preprocessing] Labels: Unsupported file type: {filename}")
        return None

    # Load image
    if filename.lower().endswith('.gif'):
        try:
            gif = imageio.mimread(image_path)
            img = np.array(gif[0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Preprocessing] Labels: Failed to read GIF: {e}")
            return None
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[Preprocessing] Labels: Failed to load image: {filename}")
        return None

    # Apply sharpening filter
    kernel = np.array([[1, -2, 1],
                       [-2, 5, -2],
                       [1, -2, 1]])
    label_image = cv2.filter2D(img, -1, kernel)

    # Super-resolution (EDSR)
    print('[Preprocessing] Labels: Beginning SR with EDSR')
    MAX_WIDTH = 1500
    MAX_HEIGHT = 500
    height, width = label_image.shape[:2]
    print(f'Width: {width}, Height: {height}')

    # Skip SR if image too large
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        print('[Preprocessing] Labels: Image too large for SR — skipping.')
        return label_image

    # Try SR upsampling
    try:
        upscaled_img = sr.upsample(label_image)
        print('[Preprocessing] Labels: SR completed successfully')
        return upscaled_img
    except Exception as e:
        print(f'[Preprocessing] Labels: SR failed: {e}')
        return label_image

