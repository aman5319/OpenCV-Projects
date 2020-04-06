import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
from PIL import Image,ImageOps

def bplot(img):
    plt.figure(figsize=(20,10))
    plt.imshow(img,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()

def cplot(img):
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def horizontal(img):

    ## Generating Horizontal Mask

    horizontal = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15,15)
    horizontalsize = horizontal.shape[1]  // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT ,(horizontalsize,1) )
    horizontal =  cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontalStructure)
    horizontal = np.where(horizontal==0,1,0)
    return horizontal

def vertical(img):

    # Generate Vertical Mask

    vertical = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0.5);
    verticalsize = vertical.shape[0] // 24
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, ( 1,verticalsize))
    vertical =  cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, verticalStructure)
    vertical = np.where(vertical==0,1,0)
    return vertical


def generateCompleteMask(horizontal , vertical):
    mask = horizontal+vertical
    mask = np.where(mask>=1 ,255,mask)
    mask = mask.astype(np.uint8)
    return mask

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255   for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def crop(img):
    gray = 255 * (img < 128).astype(np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    max_padding = 50
    x_min_padding = 0 if x < max_padding else  max_padding
    y_min_padding = 0 if y < max_padding else  max_padding
    rect = img[y-y_min_padding:y+h+max_padding, x-x_min_padding:x_min_padding+x+w] # Crop the image - with 50 padding
    return rect

def rotate(img):
    img = Image.fromarray(img)
    img = ImageOps.expand(img, border=100, fill=255)
    img = np.array(img)
    gray = cv2.bitwise_not(img)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
     
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
     
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = Image.fromarray(rotated)
    rotated = ImageOps.crop(rotated,border=110)
    return np.array(rotated)

def generateFinalImage(inputdir , outputdir):
    """
    inputdir : Path
    outputdir: Path
    """
    rho = 2
    theta = np.pi/180
    threshold = 0
    min_line_length = 0
    max_line_gap = 5
    for file in inputdir.iterdir():
        img = cv2.imread(str(file) , cv2.IMREAD_GRAYSCALE)
        img = crop(img.copy())
        img = rotate(img.copy())
        gamma = 0.111 if img.std()>= 40 else 1.0
        imgg = adjust_gamma(img.copy() , gamma)
        mask =generateCompleteMask(horizontal(imgg) , vertical(imgg))

        # Run Hough on the edge-detected image
        lines = cv2.HoughLinesP(mask, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on the image copy
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.resize(img ,  (1600,800))
        cv2.imwrite(str(outputdir/file.name) , img)

if len(sys.argv)==1:
    inputdir = Path("nachInput")
    assert inputdir.exists(), f"Make a directory with name {inputdir.name} and keep all the files into it"
    outputdir= Path("nachOutput")
    if not outputdir.exists():
        outputdir.mkdir()
    for i in outputdir.iterdir():
        i.unlink()
elif len(sys.argv)==3: 
    inputdir = Path(sys.argv[1])
    outputdir = Path(sys.argv[2])
else:
    raise Exception("Proper path is not given")
generateFinalImage(inputdir , outputdir)
