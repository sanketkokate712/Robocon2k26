import cv2
import numpy as np

img_path = '/home/sanket/robocon/models/box_custom/materials/textures/fake_kfs.png'
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is not None:
    if img.shape[2] == 4:
        print("Image has alpha channel. Removing it.")
        trans_mask = img[:,:,3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(img_path, img)
        print("Saved as RGB.")
    else:
        print("Image is already RGB.")
else:
    print("Could not read image.")
