import cv2

def CLAHE_in_space(img, color_scace, clip_limit=10, tile_grid_size=(8, 8)):
    """
    Apply histogram equalization to the image in the given color space (HSV, HLS, YUV, YCrCb) and return the equalized image in the same color space as the input image. 

    """
    color_scaces_names = ["YCrCb", "HSV", "HLS", "YUV"]
    clahe = cv2.createCLAHE(clipLimit =clip_limit , tileGridSize=tile_grid_size)
    
    if color_scace not in color_scaces_names:
        raise ValueError("Color space not supported")
    
    if color_scace == "YCrCb":
        color_scace = cv2.COLOR_RGB2YCR_CB
        inverse_color_scace = cv2.COLOR_YCR_CB2RGB
        img = cv2.cvtColor(img, color_scace)
        channels = cv2.split(img)
        equalized = clahe.apply(channels[0])
        img = cv2.merge((equalized, channels[1], channels[2]))
        
    elif color_scace == "HSV":
        color_scace = cv2.COLOR_RGB2HSV
        inverse_color_scace = cv2.COLOR_HSV2RGB
        img = cv2.cvtColor(img, color_scace)
        channels = cv2.split(img)
        equalized = clahe.apply(channels[2])
        img = cv2.merge((channels[0], channels[1], equalized))
        
        
    elif color_scace == "HLS":
        color_scace = cv2.COLOR_RGB2HLS
        inverse_color_scace = cv2.COLOR_HLS2RGB
        img = cv2.cvtColor(img, color_scace)
        channels = cv2.split(img)
        equalized = clahe.apply(channels[1])
        img = cv2.merge((channels[0], equalized, channels[2]))
        

    elif color_scace == "YUV":
        color_scace = cv2.COLOR_RGB2YUV
        inverse_color_scace = cv2.COLOR_YUV2RGB
        img = cv2.cvtColor(img, color_scace)
        channels = cv2.split(img)
        equalized = clahe.apply(channels[0])
        img = cv2.merge((equalized, channels[1], channels[2]))
 
    img = cv2.cvtColor(img, inverse_color_scace)
    
    return img

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Read in the image
    image = cv2.imread("images\CanonEOS5D\scaled-im02-raw.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Choose a color space
    color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    
    # Apply histogram equalization
    image_equalized = equalization(image, color_space)
    
    # Plot the results
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    
    ax2.imshow(image_equalized)
    ax2.set_title('Equalized Image', fontsize=50)
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    