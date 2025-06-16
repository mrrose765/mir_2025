# Defintion de toute les fonctions à appeller dans l'interface
import json
import os

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QMessageBox
from skimage import img_as_ubyte
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
from skimage.transform import resize

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def showDialog():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    returnValue = msgBox.exec()


def generateHistogramme_HSV(features_folder, image_paths, progressBar=None):
    if not os.path.isdir(f"{features_folder}/HSV"):
        os.mkdir(f"{features_folder}/HSV")
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img], [0], None, [180], [0, 180])
        histS = cv2.calcHist([img], [1], None, [256], [0, 256])
        histV = cv2.calcHist([img], [2], None, [256], [0, 256])
        feature = np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt(f"{features_folder}/HSV/" + str(num_image) + ".txt", feature)

        if progressBar:
            progressBar.setValue(100 * ((i + 1) / total))
    print("Indexation Hist HSV terminée !!!!")


def generateHistogramme_Color(features_folder, image_paths, progressBar):
    if not os.path.isdir(f"{features_folder}/BGR"):
        os.mkdir(f"{features_folder}/BGR")
    total = len(image_paths)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        histB = cv2.calcHist([img], [0], None, [256], [0, 256])
        histG = cv2.calcHist([img], [1], None, [256], [0, 256])
        histR = cv2.calcHist([img], [2], None, [256], [0, 256])
        feature = np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt(f"{features_folder}/BGR/" + str(num_image) + ".txt", feature)

        if progressBar:
            progressBar.setValue(100 * ((i + 1) / total))
    print("indexation Hist Couleur terminée !!!!")


def generateSIFT(features_folder, image_paths, progressBar=None):
    if not os.path.isdir(f"{features_folder}/SIFT"):
        os.mkdir(f"{features_folder}/SIFT")
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
        np.savetxt(f"{features_folder}/SIFT/" + str(num_image) + ".txt", des)

        if progressBar:
            progressBar.setValue(100 * ((i + 1) / total))
    print("Indexation SIFT terminée !!!!")


def generateORB(features_folder, image_paths, progressBar=None):
    if not os.path.isdir(f"{features_folder}/ORB"):
        os.mkdir(f"{features_folder}/ORB")
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
        np.savetxt(f"{features_folder}/ORB/" + str(num_image) + ".txt", descrip1)

        if progressBar:
            progressBar.setValue(100 * ((i + 1) / total))
    print("Indexation ORB terminée !!!!")


def generateGLCM(features_folder, image_paths, progressBar=None):
    if not os.path.isdir(f"{features_folder}/GLCM"):
        os.mkdir(f"{features_folder}/GLCM")
    distances = [1, -1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
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
        np.savetxt(f"{features_folder}/GLCM/" + str(num_image) + ".txt", feature)

        if progressBar:
            progressBar.setValue(100 * ((i + 1) / total))
    print("Indexation GLCM terminée !!!!")


def generateLBP(features_folder, image_paths, progressBar=None):
    if not os.path.isdir(f"{features_folder}/LBP"):
        os.mkdir(f"{features_folder}/LBP")
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
        for k in range(int(fullLBPmatrix.shape[0] / subSize[0])):
            for j in range(int(fullLBPmatrix.shape[1] / subSize[1])):
                subVector = fullLBPmatrix[k * subSize[0]:(k + 1) * subSize[0],
                            j * subSize[1]:(j + 1) * subSize[1]].ravel()
                subHist, _ = np.histogram(subVector, bins=int(2 ** points), range=(0, 2 ** points))
                histograms = np.concatenate((histograms, subHist), axis=None)

        num_image = os.path.splitext(os.path.basename(image_path))[0]
        np.savetxt(f"{features_folder}/LBP/" + str(num_image) + ".txt", histograms)

        if progressBar:
            progressBar.setValue(100 * ((i + 1) / total))
    print("Indexation LBP terminée !!!!")


def generateHOG(features_folder, filenames, progressBar):
    if not os.path.isdir(f"{features_folder}/HOG"):
        os.mkdir(f"{features_folder}/HOG")

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
            np.savetxt(f"{features_folder}/HOG/{filename}.txt", features)

            progressBar.setValue(int(100 * (i + 1) / total))
            i += 1

        except Exception as e:
            print(f"Erreur sur {path} : {e}")

    print("Indexation HOG terminée !!!!")

def generateMobileNetFeatures(features_folder, filenames, progressBar):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger MobileNetV2 et enlever la dernière couche de classification
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Créer le dossier de sortie si besoin
    output_folder = f"{features_folder}/MobileNet"
    os.makedirs(output_folder, exist_ok=True)

    total = len(filenames)

    for i, path in enumerate(filenames):
        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(tensor)

            features_np = features.cpu().numpy().squeeze()
            filename = os.path.splitext(os.path.basename(path))[0]
            np.savetxt(f"{output_folder}/{filename}.txt", features_np, fmt="%.6f")

            if progressBar:
                progressBar.setValue(int(100 * (i + 1) / total))

        except Exception as e:
            print(f"Erreur sur {path} : {e}")

    print("Indexation MobileNet terminée !!!")


def extractReqFeatures(fileName, algo_choice, model=None, transform=None, device=None):
    print(algo_choice)
    if fileName:
        img = cv2.imread(fileName)
        if img is None:
            print(f"Erreur de lecture de l'image : {fileName}")
            return None

        if algo_choice == 1:  # Couleurs
            histB = cv2.calcHist([img], [0], None, [256], [0, 256])
            histG = cv2.calcHist([img], [1], None, [256], [0, 256])
            histR = cv2.calcHist([img], [2], None, [256], [0, 256])
            vect_features = np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)

        elif algo_choice == 2:  # Histo HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            histS = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            histV = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            vect_features = np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)

        elif algo_choice == 3:  # SIFT
            sift = cv2.SIFT_create()  # cv2.xfeatures2d.SIFT_create() pour py < 3.4
            # Find the key point
            _, vect_features = sift.detectAndCompute(img, None)
            if vect_features is None:
                vect_features = np.zeros((1,128))

        elif algo_choice == 4:  # ORB
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            _, vect_features = orb.detectAndCompute(img, None)
            if vect_features is None:
                vect_features = np.zeros((1,32))

        elif algo_choice == 5:  # GLCM
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = img_as_ubyte(gray)
            distances = [1, -1]
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True)
            glcmProperties = []
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in props:
                glcmProperties.append(greycoprops(glcmMatrix, prop).ravel())
            vect_features = np.concatenate(glcmProperties, axis=None)

        elif algo_choice == 6:  # LBP
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(gray, (350, 350))
            points, radius = 8, 1
            method = 'default'
            subSize = (70, 70)
            lbp_matrix = local_binary_pattern(img_resized, points, radius, method)
            histograms = []
            for k in range(int(lbp_matrix.shape[0] / subSize[0])):
                for j in range(int(lbp_matrix.shape[1] / subSize[1])):
                    subVector = lbp_matrix[k * subSize[0]:(k + 1) * subSize[0],
                                           j * subSize[1]:(j + 1) * subSize[1]].ravel()
                    subHist, _ = np.histogram(subVector, bins=int(2 ** points), range=(0, 2 ** points))
                    histograms = np.concatenate((histograms, subHist), axis=None)
            vect_features = histograms

        elif algo_choice == 7:  # HOG
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            vect_features, _ = hog(resized,
                                   orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   block_norm='L2-Hys',
                                   visualize=True,
                                   feature_vector=True)
            
        elif algo_choice == 8:  # MobileNetV2
            if model is None or transform is None or device is None:
                raise ValueError("Model, transform, and device must be provided for MobileNet feature extraction.")
            image_pil = Image.open(fileName).convert("RGB")
            tensor = transform(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(tensor)
            vect_features = features.cpu().numpy().squeeze()

        elif algo_choice == 9:  # ViT-21k
            # In this case, we can only use the images from the database. (Already indexed)
            # Load features from the indexed files
            features_path = "features/image_features/ViT-21k/"
            image_name = os.path.splitext(os.path.basename(fileName))[0] + ".txt"
            full_path = os.path.join(features_path, image_name)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Le fichier de caractéristiques pour l'image {fileName} n'existe pas dans la base de données.")
            vect_features = np.loadtxt(full_path)
        else:
            raise ValueError(f"Descripteur inconnu: {algo_choice}")

        # np.savetxt("Methode_" + str(algo_choice) + "_requete.txt", vect_features)
        # print("saved")
        # print("vect_features", vect_features)
        return vect_features


def _normalize_vector(v):
    """
    Normalise un vecteur v en le divisant par sa norme.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def concat_vectors(img_feat, txt_feat, normalize=True):
    if normalize:
        img_feat = _normalize_vector(img_feat)
        txt_feat = _normalize_vector(txt_feat)

    concat = np.concatenate([img_feat, txt_feat])
    return concat


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_device():
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("PyTorch is not installed. Defaulting to CPU.")
        return "cpu"

def extract_class(name):
  """
  Les images sont nommées comme : A_B_animal_race-X où
    A = animal_id
    B = race_id
  L'id de la classe sera: A*6+B
  Ex:
    0_5_araignees_tarantula_795 (5)
    2_2_oiseaux_greatgreyowl_2092 (2*6+2 = 14)
  """
  return int(name.split("_")[0])*6 + int(name.split("_")[1])