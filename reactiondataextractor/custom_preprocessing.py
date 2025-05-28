import cv2
import numpy as np
from cv2 import dnn_superres
from processors import ImageReader, ImageScaler, ImageNormaliser, Binariser

def preprocess_for_arrows(image_path: str):
    print(f"[Preprocessing] Arrows: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    fig = ImageNormaliser(fig).process()
    fig = Binariser(fig).process()

    print('[Preprocessing] Arrows: Beginning edge-focused sharpening')
    kernel = np.array([[1, -2, 1],
                       [-2, 5, -2],
                       [1, -2, 1]])
    
    # Apply sharpening filter to the image
    sharpened_img = cv2.filter2D(fig.img, -1, kernel)
    fig.img = sharpened_img

    print("[Preprocessing] Arrows: Done")
    return fig

def preprocess_for_diagrams(image_path: str):
    print(f"[Preprocessing] Diagrams: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.RGB).process()
    fig = ImageScaler(fig, resize_min_dim_to=2048).process()
    fig = ImageNormaliser(fig).process()
    fig = Binariser(fig).process()

    print('[Preprocessing] Diagrams: Beginning moderate sharpening')
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply sharpening filter to the image inside the Figure object
    sharpened_img = cv2.filter2D(fig.img, -1, kernel)
    fig.img = sharpened_img  # Replace the image inside the Figure object

    print("[Preprocessing] Diagrams: Done")
    return fig

def preprocess_for_conditions(image_path: str):
    print(f"[Preprocessing] Conditions: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    fig = ImageNormaliser(fig).process()
    fig = Binariser(fig).process()
    print("[Preprocessing] Conditions: Done")
    return fig

def preprocess_for_labels(image_path: str):
    print(f"[Preprocessing] Labels: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    fig = ImageNormaliser(fig).process()
    fig = Binariser(fig).process()

    print('[Preprocessing] Labels: Beginning edge-focused sharpening')
    kernel = np.array([[1, -2, 1],
                       [-2, 5, -2],
                       [1, -2, 1]])
    
    # Apply sharpening filter to the image inside the Figure object
    sharpened_img = cv2.filter2D(fig.img, -1, kernel)
    fig.img = sharpened_img  # Replace the image inside the Figure object

    print('[Preprocessing] Labels: Beginning SR with EDSR')
    MAX_WIDTH = 1655
    MAX_HEIGHT = 658

    height, width = fig.img.shape[:2]

    sr_model_path = 'EDSR_x2.pb'
    sr = dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel(sr_model_path)
        sr.setModel('edsr', 2)
    except Exception as e:
        print(f'[Preprocessing] Labels: ERROR: Failed to load SR model: {e}')
        sys.exit(1) # Abort

    if width > MAX_WIDTH or height > MAX_HEIGHT:
        print('[Preprocessing] Labels: Image too large for SR — skipping SR.')
        return fig

    try:
        upscaled_img = sr.upsample(fig.img)
        fig.img = upscaled_img  # Replace with upsampled image
        print('[Preprocessing] Labels: SR completed successfully')
    except Exception as e:
        print(f'[Preprocessing] Labels: SR failed: {e}')

    return fig
