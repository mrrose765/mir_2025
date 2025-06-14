from PyQt5 import QtWidgets, QtCore, QtGui
import os
import functions as f
import time
import numpy as np
import math
import cv2
import distances as d
import matplotlib.pyplot as plt
from metrics import precision_at_k, recall, average_precision, r_precision, mean_average_precision


class TabRechercheController:
    def __init__(self, main_app):
        self.ui = main_app

        # Liens des boutons pour recherche
        self.ui.comboBox_top.addItems(["20", "50"])
        self.ui.quitter_rech.clicked.connect(self.ui.Quitter)
        self.ui.charger_rech.clicked.connect(self.OuvrirImage)
        self.ui.charger_desc.clicked.connect(self.loadFeatures)
        self.ui.chercher.clicked.connect(self.Recherche)
        self.ui.calcul_RP.clicked.connect(self.rappel_precision)
        self.ui.pushButton.clicked.connect(self.afficher_metriques)


    def OuvrirImage(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpeg *.jpg *.bmp)"
        )
        if self.fileName:
            pixmap = QtGui.QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.ui.label_requete.width(),
                                   self.ui.label_requete.height(), QtCore.Qt.KeepAspectRatio)
            self.ui.label_requete.setPixmap(pixmap)
            self.ui.label_requete.setAlignment(QtCore.Qt.AlignCenter)
        else:
            print("Aucune image requête sélectionnée. Dans ouvrir")

    def loadFeatures(self):

        self.algo_choice = 0
        folder_model = self.ui.features_folder  + "/image_features"

        # Sélection du dossier selon le descripteur
        if self.ui.checkBox_HistC_rech.isChecked():
            folder_model += '/BGR'
            self.algo_choice = 1
        elif self.ui.checkBox_HSV_rech.isChecked():
            folder_model += '/HSV'
            self.algo_choice = 2
        elif self.ui.checkBox_SIFT_rech.isChecked():
            folder_model += '/SIFT'
            self.algo_choice = 3
        elif self.ui.checkBox_ORB_rech.isChecked():
            folder_model += '/ORB'
            self.algo_choice = 4
        elif self.ui.checkBox_GLCM_rech.isChecked():
            folder_model += '/GLCM'
            self.algo_choice = 5
        elif self.ui.checkBox_LBP_rech.isChecked():
            folder_model += '/LBP'
            self.algo_choice = 6
        elif self.ui.checkBox_HOG_rech.isChecked():
            folder_model += '/HOG'
            self.algo_choice = 7
        elif self.ui.checkBox_Moments_rech.isChecked():
            folder_model += '/MOMENTS'
            self.algo_choice = 8
        else:
            print("Merci de sélectionner un descripteur")
            self.ui.showDialog()
            return

        # Nettoyer la grille d'affichage
        for i in reversed(range(self.ui.gridLayout.count())):
            self.ui.gridLayout.itemAt(i).widget().setParent(None)

        # Mise à jour des distances
        if self.algo_choice in [3, 4, 5, 6, 7, 8]:
            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(["Brute force", "Flann"])
        else:
            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(["Euclidienne", "Correlation", "Chi carre", "Intersection", "Bhattacharyya"])

        # Vérification image requête
        if not self.fileName:
            print("Merci de charger une image avec le bouton Ouvrir")
            return

        # Chargement
        print("Chargement descripteurs en cours ...")
        start_time = time.time()
        self.features1 = []
        pas = 0

        # Construire un index de toutes les images dans imgDB
        image_index = {}
        for root, _, files in os.walk(self.ui.image_folder):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_index[os.path.splitext(f)[0]] = os.path.join(root, f)

        # Charger les features en cherchant l'image correspondante dans l'index
        all_txt = []
        for root, _, files in os.walk(folder_model):
            for file in files:
                if file.endswith(".txt"):
                    all_txt.append(os.path.join(root, file))

        total_files = len(all_txt)

        for txt_path in all_txt:
            feature = np.loadtxt(txt_path)
            base_name = os.path.splitext(os.path.basename(txt_path))[0]

            if base_name in image_index:
                img_path = image_index[base_name]
                self.features1.append((img_path, feature))
            else:
                print(f"[AVERTISSEMENT] Image non trouvée pour : {base_name}")

            pas += 1
            self.ui.progressBar_rech.setValue(int(100 * (pas / total_files)))

        end_time = time.time()
        print(f"Temps de chargement du descripteur {folder_model} : {end_time - start_time:.4f} secondes")

    def Recherche(self):
        for i in reversed(range(self.ui.gridLayout.count())):
            self.ui.gridLayout.itemAt(i).widget().setParent(None)

        if self.algo_choice == 0:
            print("Il faut choisir une méthode !")
            return

        print(f"[Recherche] Image en mémoire : {getattr(self, 'fileName', 'Non définie')}")
        if not hasattr(self, 'fileName') or not self.fileName:
            print("Aucune image requête sélectionnée.")
            return

        start_time = time.time()
        print("Extraction descripteur image requête...")

        req = f.extractReqFeatures(self.fileName, self.algo_choice)
        self.sortie = int(self.ui.comboBox_top.currentText())
        distanceName = self.ui.comboBox.currentText()

        voisins = d.getkVoisins(self.features1, req, self.sortie, distanceName)

        self.path_image_plus_proches = []
        self.nom_image_plus_proches = []

        for k in range(self.sortie):
            self.path_image_plus_proches.append(voisins[k][0])
            self.nom_image_plus_proches.append(os.path.basename(voisins[k][0]))

        col = 3
        target_width = 150
        target_height = 150
        for i in range(math.ceil(self.sortie / col)):
            for j in range(col):
                idx = i * col + j
                if idx >= len(self.path_image_plus_proches):
                    break
                img = cv2.imread(self.path_image_plus_proches[idx], 1)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                qImg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)

                # Création du QLabel
                label = QtWidgets.QLabel()
                label.setPixmap(pixmap)
                label.setFixedSize(target_width, target_height)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setStyleSheet("padding:5px; border: 1px solid #ccc;")

                self.ui.gridLayout.addWidget(label, i, j)

        end_time = time.time()
        print(f"Temps de recherche pour le descripteur {self.algo_choice} : {end_time - start_time:.4f} secondes")

    def rappel_precision(self):
        start_time = time.time()

        rappel_precision = []
        rappels = []
        precisions = []

        filename_req = os.path.basename(self.fileName)
        parts_req = filename_req.split("_")
        if len(parts_req) < 5:
            print(f"Nom de fichier invalide pour la requête : {filename_req}")
            return
        classe_image_requete = parts_req[2] + "_" + parts_req[3]

        for j in range(self.sortie):
            filename_proche = os.path.basename(self.path_image_plus_proches[j])
            parts_proche = filename_proche.split('_')
            if len(parts_proche) < 5:
                print(f"Nom de fichier invalide : {filename_proche}")
                rappel_precision.append(0)
                continue

            classe_image_proche = parts_proche[2] + "_" + parts_proche[3]

            if classe_image_requete == classe_image_proche:
                rappel_precision.append(1)
            else:
                rappel_precision.append(0)

        total_pertinents = sum(rappel_precision)

        for i in range(self.sortie):
            val = sum(rappel_precision[:i + 1])
            precision = val / (i + 1)
            rappel = val / total_pertinents if total_pertinents > 0 else 0
            rappels.append(rappel)
            precisions.append(precision)

        print(rappels)
        print(precisions)

        # Création de la courbe R/P
        plt.plot(rappels, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("R/P" + str(self.sortie) + " voisins de l'image")

        # Enregistrement de la courbe RP
        save_folder = os.path.join(".", "rp_courbes")
        os.makedirs(save_folder, exist_ok=True)
        image_name = os.path.splitext(filename_req)[0]
        save_path = os.path.join(save_folder, f"{image_name}_rp.png")
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()

        # Affichage dans le QLabel
        img = cv2.imread(save_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)

        label_width = self.ui.label_requete.frameGeometry().width()
        label_height = self.ui.label_requete.frameGeometry().height()
        self.ui.label_courbe.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_courbe.setPixmap(
            pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        end_time = time.time()
        print(f"Temps de calcul de la courbe R/P: {end_time - start_time:.4f} secondes")

    def afficher_metriques(self):
        if not hasattr(self, "nom_image_plus_proches") or not self.nom_image_plus_proches:
            self.ui.label_metriques.setText("Aucun résultat de recherche.")
            return

        # Image requête
        filename_req = os.path.basename(self.fileName)

        # Classe de l'image requête (à partir du nom)
        classe_requete = "_".join(filename_req.split("_")[2:4])  # e.g., 'araignees_tarantula'

        # Images pertinentes = celles de la même classe
        relevant = [
            img for img in self.nom_image_plus_proches
            if "_".join(img.split("_")[2:4]) == classe_requete
        ]

        # Liste des résultats retournés
        retrieved = self.nom_image_plus_proches

        # k = nombre de résultats
        k = len(retrieved)

        # Calcul des métriques
        p_at_k = precision_at_k(retrieved, relevant, k)
        rec = recall(retrieved, relevant)
        ap = average_precision(retrieved, relevant)
        rprec = r_precision(retrieved, relevant)
        map_ = mean_average_precision([retrieved], [relevant])

        # Affichage
        texte = f"""\
        Précision@{k} : {p_at_k:.3f}
        Rappel : {rec:.3f}
        Moyenne Précision : {ap:.3f}
        MAP : {map_:.3f}
        R-Précision : {rprec:.3f}"""

        self.ui.label_metriques.setText(texte)