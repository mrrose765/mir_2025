#Defintion de toute les fonctions à appeller dans l'interface
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern

def showDialog():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    returnValue = msgBox.exec()

def generateHistogramme_HSV(image_paths, progressBar=None):
    if not os.path.isdir("HSV"):
        os.mkdir("HSV")
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img],[0],None,[180],[0,180])
        histS = cv2.calcHist([img],[1],None,[256],[0,256])
        histV = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt("HSV/"+str(num_image)+".txt", feature)

        if progressBar:
            progressBar.setValue(100*((i+1)/total))
    print("Indexation Hist HSV terminée !!!!")
        
def generateHistogramme_Color(image_paths, progressBar):
    if not os.path.isdir("BGR"):
        os.mkdir("BGR")
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        histB = cv2.calcHist([img],[0],None,[256],[0,256])
        histG = cv2.calcHist([img],[1],None,[256],[0,256])
        histR = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt("BGR/"+str(num_image)+".txt" ,feature)

        if progressBar:
            progressBar.setValue(100*((i+1)/total))
    print("indexation Hist Couleur terminée !!!!")

def generateSIFT(image_paths, progressBar=None):
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")
    total = len(image_paths)
    sift = cv2.SIFT_create()
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        kps, des = sift.detectAndCompute(img, None)
        if des is None:
            des = np.array([])  # pour éviter erreur sauvegarde

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt("SIFT/"+str(num_image)+".txt", des)

        if progressBar:
            progressBar.setValue(100*((i+1)/total))
    print("Indexation SIFT terminée !!!!")


def generateORB(image_paths, progressBar=None):
    if not os.path.isdir("ORB"):
        os.mkdir("ORB")
    total = len(image_paths)
    orb = cv2.ORB_create()
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        key_point1, descrip1 = orb.detectAndCompute(img, None)
        if descrip1 is None:
            descrip1 = np.array([])

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt("ORB/"+str(num_image)+".txt", descrip1)

        if progressBar:
            progressBar.setValue(100*((i+1)/total))
    print("Indexation ORB terminée !!!!")

def generateGLCM(image_paths, progressBar=None):
    if not os.path.isdir("GLCM"):
        os.mkdir("GLCM")
    distances = [1, -1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True)
        glcmProperties = []
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            glcmProperties.append(greycoprops(glcmMatrix, prop).ravel())
        feature = np.concatenate(glcmProperties, axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt("GLCM/"+str(num_image)+".txt", feature)

        if progressBar:
            progressBar.setValue(100*((i+1)/total))
    print("Indexation GLCM terminée !!!!")

def generateLBP(image_paths, progressBar=None):
    if not os.path.isdir("LBP"):
        os.mkdir("LBP")
    points = 8
    radius = 1
    method = 'default'
    subSize = (70, 70)
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (350, 350))
        fullLBPmatrix = local_binary_pattern(img, points, radius, method)
        histograms = []
        for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
            for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0], j*subSize[1]:(j+1)*subSize[1]].ravel()
                subHist, _ = np.histogram(subVector, bins=int(2**points), range=(0, 2**points))
                histograms = np.concatenate((histograms, subHist), axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt("LBP/"+str(num_image)+".txt", histograms)

        if progressBar:
            progressBar.setValue(100*((i+1)/total))
    print("Indexation LBP terminée !!!!")

def generateHOG(filenames, progressBar):
    if not os.path.isdir("HOG"):
        os.mkdir("HOG")
    
    i = 0
    total = len(filenames)

    for path in filenames:
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Erreur de lecture: {path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))  # taille standardisée

            features, hog_image = hog(resized,
                                      orientations=9,
                                      pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2),
                                      block_norm='L2-Hys',
                                      visualize=True,
                                      feature_vector=True)

            filename = os.path.basename(path).split('.')[0]
            np.savetxt(f"HOG/{filename}.txt", features)

            progressBar.setValue(int(100 * (i + 1) / total))
            i += 1

        except Exception as e:
            print(f"Erreur sur {path} : {e}")

    print("Indexation HOG terminée !!!!")
	
def extractReqFeatures(fileName,algo_choice):  
    print(algo_choice)
    if fileName : 
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))
            
        if algo_choice==1: #Couleurs
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif algo_choice==2: # Histo HSV
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        elif algo_choice==3: #SIFT
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
    
        elif algo_choice==4: #ORB
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)
			
        np.savetxt("Methode_"+str(algo_choice)+"_requete.txt" ,vect_features)
        print("saved")
        #print("vect_features", vect_features)
        return vect_features