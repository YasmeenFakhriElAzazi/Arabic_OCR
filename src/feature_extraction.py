import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pytesseract

# Function to get immediate subdirectories of a given path
def get_immediate_subdirectories(path):
    return [f.name for f in os.scandir(path) if f.is_dir()]

# Function to calculate the white-to-black pixel ratio in an image
def whiteBlackRatio(img):
    black_pixels = np.sum(img == 0)  # Count black pixels
    white_pixels = np.sum(img == 255)  # Count white pixels
    return white_pixels / black_pixels if black_pixels != 0 else 0

# Function to count the number of black pixels in the image
def blackPixelsCount(img):
    return np.sum(img == 0)  # Count black pixels

# Function to count the number of horizontal transitions in the image
def horizontalTransitions(img):
    transitions = np.sum(img[:-1, :] != img[1:, :])  # Check for changes between rows
    return transitions

# Function to count the number of vertical transitions in the image
def verticalTransitions(img):
    transitions = np.sum(img[:, :-1] != img[:, 1:])  # Check for changes between columns
    return transitions

# Function to extract various features from the image
def getFeatures(img):
    y, x = img.shape
    featuresList = []
    
    # Extract height-to-width ratio
    featuresList.append(y / x)
    
    # Extract white-to-black ratio
    featuresList.append(whiteBlackRatio(img))
    
    # Extract horizontal and vertical transitions
    featuresList.append(horizontalTransitions(img))
    featuresList.append(verticalTransitions(img))
    
    # Divide the image into 4 quadrants and compute their white-to-black ratios
    topLeft = img[:y // 2, :x // 2]
    topRight = img[:y // 2, x // 2:]
    bottomLeft = img[y // 2:, :x // 2]
    bottomRight = img[y // 2:, x // 2:]
    
    featuresList.extend([whiteBlackRatio(topLeft), whiteBlackRatio(topRight),
                         whiteBlackRatio(bottomLeft), whiteBlackRatio(bottomRight)])
    
    # Black pixel ratio between quadrants
    topLeftCount = blackPixelsCount(topLeft)
    topRightCount = blackPixelsCount(topRight)
    bottomLeftCount = blackPixelsCount(bottomLeft)
    bottomRightCount = blackPixelsCount(bottomRight)
    
    featuresList.extend([topLeftCount / topRightCount if topRightCount != 0 else 0,
                         bottomLeftCount / bottomRightCount if bottomRightCount != 0 else 0,
                         topLeftCount / bottomLeftCount if bottomLeftCount != 0 else 0,
                         topRightCount / bottomRightCount if bottomRightCount != 0 else 0,
                         topLeftCount / bottomRightCount if bottomRightCount != 0 else 0,
                         topRightCount / bottomLeftCount if bottomLeftCount != 0 else 0])
    
    # Calculate the center of mass (centroid) and horizontal distribution
    xCenter, yCenter = np.mean(np.where(img == 255), axis=1) if np.sum(img == 255) != 0 else (0, 0)
    featuresList.append(xCenter)
    featuresList.append(yCenter)
    
    return featuresList

# Function to get all files in a directory, recursively
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles.extend(getListOfFiles(fullPath))
        else:
            allFiles.append(fullPath)
    return allFiles

# Function to train and classify data using SVM
def trainAndClassify(data, classes):
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.20)
    svclassifier = SVC(kernel="rbf", gamma=0.005, C=1000)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Function to remove margins from the image
def removeMargins(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

# Function to extract words from an image using pytesseract and get features
def extract_words_and_get_features(img, save=False, cut=3, padding=10, output_dir="output_segments"):
    # Use pytesseract to extract words
    text = pytesseract.image_to_string(img)
    words = text.split()  # Split text into words
    
    data_list = []
    classes_list = []
    
    # For each word, extract its features
    for word in words:
        word_img = img  # Placeholder, replace with actual word segmentation logic
        features = getFeatures(word_img)
        
        data_list.append(features)
        classes_list.append(word)  # Using the word as the class label (or other logic)
    
    return np.array(data_list), np.array(classes_list)

# Main function to execute the entire process
def main():
    data = []
    classes = []
    directory = "../LettersDataset"
    
    chars = get_immediate_subdirectories(directory)
    count = 0
    numOfFeatures = 16  # Number of features to be extracted
    
    charPositions = ["Beginning", "End", "Isolated", "Middle"]
    
    for char in chars:
        for position in charPositions:
            if os.path.isdir(directory + "/" + char + "/" + position):
                listOfFiles = getListOfFiles(directory + "/" + char + "/" + position)
                for filename in listOfFiles:
                    img = cv.imread(filename)
                    cropped = removeMargins(img)
                    gray_img = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
                    binary_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)[1]  # Simple binary threshold
                    
                    word_data, word_classes = extract_words_and_get_features(binary_img)
                    data.extend(word_data)
                    classes.extend(word_classes)
                    count += len(word_classes)
    
    data = np.array(data)
    classes = np.array(classes)
    
    trainAndClassify(data, classes)

# Execute the program
main()
