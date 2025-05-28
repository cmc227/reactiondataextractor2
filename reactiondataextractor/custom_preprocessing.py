from processors import ImageReader, ImageScaler, ImageNormaliser, Binariser

def preprocess_for_arrows(image_path: str):
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    fig = ImageNormaliser(fig).process()
    fig = Binariser(fig).process()
    return fig

def preprocess_for_diagrams(image_path: str):
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.RGB).process()
    fig = ImageScaler(fig, resize_min_dim_to=2048).process()
    fig = ImageNormaliser(fig).process()
    return fig

def preprocess_for_labels(image_path: str):
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    return fig

def preprocess_for_conditions(image_path: str):
    fig = ImageReader(image_path, color_mode=ImageReader.COLOR_MODE.GRAY).process()
    fig = ImageScaler(fig, resize_min_dim_to=1024).process()
    return fig
