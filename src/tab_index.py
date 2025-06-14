from PyQt5 import QtWidgets, QtCore, QtGui
import os
import functions as f



class TabIndexController:
    def __init__(self, main_app):
        self.ui = main_app

        # Liens des boutons pour indexation
        self.ui.charger.clicked.connect(self.Ouvrir_index)
        self.ui.quitter.clicked.connect(self.ui.Quitter)
        self.ui.tableView.clicked.connect(self.CliquerTab)
        self.ui.indexer.clicked.connect(self.extractFeatures)

    def Ouvrir_index(self):
        """Fonction pour ouvrir un explorateur de fichiers et charger les images"""

        self.list_images_index = []
        self.Dossier_images_index = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select directory', "C://",
                                                                               QtWidgets.QFileDialog.ShowDirsOnly) + "/"

        extensions_images = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')
        for root, dirs, files in os.walk(self.Dossier_images_index):
            for file in files:
                if file.lower().endswith(extensions_images):
                    chemin_complet = os.path.join(root, file)
                    self.list_images_index.append(chemin_complet)

        # Affichage de la première image
        if self.list_images_index:
            pixmap = QtGui.QPixmap(self.list_images_index[0])
            pixmap = pixmap.scaled(self.ui.image.width(),
                                   self.ui.image.height(),
                                   QtCore.Qt.KeepAspectRatio)
            self.ui.image.setPixmap(pixmap)
            self.ui.image.setAlignment(QtCore.Qt.AlignCenter)

        # Remplir le tableView avec les noms de fichiers
        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(["File name"])

        for chemin_image in self.list_images_index:
            nom_fichier = os.path.basename(chemin_image)
            item = QtGui.QStandardItem(nom_fichier)
            item.setEditable(False)
            item.setData(chemin_image, QtCore.Qt.UserRole)
            model.appendRow([item])

        self.ui.tableView.setModel(model)

    def CliquerTab(self):
        index = self.ui.tableView.selectionModel().currentIndex()
        chemin_complet = index.data(QtCore.Qt.UserRole)  # Récupère le chemin complet
        if chemin_complet and os.path.exists(chemin_complet):
            pixmap = QtGui.QPixmap(chemin_complet)
            pixmap = pixmap.scaled(self.ui.image.width(),
                                   self.ui.image.height(), QtCore.Qt.KeepAspectRatio)
            self.ui.image.setPixmap(pixmap)
            self.ui.image.setAlignment(QtCore.Qt.AlignCenter)

    def extractFeatures(self):
        # Vérifie que des images sont bien chargées
        if not hasattr(self, 'list_images_index') or len(self.list_images_index) < 1:
            print("Merci de charger la base de données avec le bouton Ouvrir")
            return

        # Vérifie que l'utilisateur a coché au moins un descripteur
        if not (self.ui.checkBox_HistC_index.isChecked() or self.ui.checkBox_HSV_index.isChecked() or
                self.ui.checkBox_SIFT_index.isChecked() or self.ui.checkBox_ORB_index.isChecked() or
                self.ui.checkBox_GLCM_index.isChecked() or self.ui.checkBox_LBP_index.isChecked() or
                self.ui.checkBox_HOG_index.isChecked()):
            print("Merci de sélectionner un descripteur via le menu ...")
            f.showDialog()
            return
        image_features_folder = self.ui.features_folder + "/image_features"

        # Exécute les descripteurs cochés
        if self.ui.checkBox_HistC_index.isChecked():
            f.generateHistogramme_Color(image_features_folder, self.list_images_index, self.ui.progressBar)

        if self.ui.checkBox_HSV_index.isChecked():
            f.generateHistogramme_HSV(image_features_folder, self.list_images_index, self.ui.progressBar)

        if self.ui.checkBox_SIFT_index.isChecked():
            f.generateSIFT(image_features_folder, self.list_images_index, self.ui.progressBar)

        if self.ui.checkBox_ORB_index.isChecked():
            f.generateORB(image_features_folder, self.list_images_index, self.ui.progressBar)

        if self.ui.checkBox_GLCM_index.isChecked():
            f.generateGLCM(image_features_folder, self.list_images_index, self.ui.progressBar)

        if self.ui.checkBox_LBP_index.isChecked():
            f.generateLBP(image_features_folder, self.list_images_index, self.ui.progressBar)

        if self.ui.checkBox_HOG_index.isChecked():
            f.generateHOG(image_features_folder, self.list_images_index, self.ui.progressBar)

        print("Indexation terminée.")