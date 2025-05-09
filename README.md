# Arabic_OCR

preprocessing functions :

binary_otsus(image_url, apply_blur= True, apply_morph= True) // Converts an input image to a binary image (black and white).

deskew(binary_img)

vexpand(gray_img, pad: int = 10, color: int = 255)
//Add vertical padding (top and bottom) to a grayscale image.

hexpand(gray_img, pad: int = 10, color: int = 255)
//Add horizontal padding (left and right) to a grayscale image.

connectedComponents(img)
//Labels connected regions of non-zero pixels with unique integers to Isolate individual characters or words
