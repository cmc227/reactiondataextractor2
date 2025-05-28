from processors import ImageReader, ImageScaler, ImageNormaliser, Binariser

def preprocess_for_arrows(image_path: str):
    print(f"[Preprocessing] Arrows: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    fig = ImageNormaliser(fig).process()
    fig = Binariser(fig).process()
    print("[Preprocessing] Arrows: Done")
    return fig

def preprocess_for_diagrams(image_path: str):
    print(f"[Preprocessing] Diagrams: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.RGB).process()
    fig = ImageScaler(fig, resize_min_dim_to=2048).process()
    fig = ImageNormaliser(fig).process()
    print("[Preprocessing] Diagrams: Done")
    return fig

def preprocess_for_labels(image_path: str):
    print(f"[Preprocessing] Labels: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    print("[Preprocessing] Labels: Done")
    return fig

def preprocess_for_conditions(image_path: str):
    print(f"[Preprocessing] Conditions: Loading and processing image {image_path}")
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    print("[Preprocessing] Conditions: Done")
    return fig
