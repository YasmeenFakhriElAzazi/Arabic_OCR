import numpy as np
import cv2 as cv
import multiprocessing as mp


def save_image(img, folder, title):
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    process_name = mp.current_process().name
    procees_number = ""
    if process_name and process_name.find("-") != -1:
        procees_number = process_name.split("-")[1]

    img_path = f"./{folder}/p{procees_number}_{title}.png"
    cv.imwrite(img_path, img)
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    return img_path


def projection(gray_img, axis: str = "horizontal"):
    """Compute the horizontal or the vertical projection of a gray image"""

    if axis == "horizontal":
        projection_bins = np.sum(gray_img, 1).astype("int32")
    elif axis == "vertical":
        projection_bins = np.sum(gray_img, 0).astype("int32")

    return projection_bins