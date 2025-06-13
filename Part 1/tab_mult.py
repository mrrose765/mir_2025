import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import os
import operator
import math
import cv2

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_k_neighbors(features_dict, query_feature, k):
    distances = []
    for name, features in features_dict.items():
        sim = cosine_similarity(query_feature, features)
        dist = 1 - sim
        distances.append((name, dist))
    # Sort distances in descending order
    distances.sort(key=operator.itemgetter(1))
    # Return the k closest features
    return distances[:k]

def load_json_file(file_path):
    """
    Load a JSON file and return its content.
    """
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class TabMultimodalController:
    def __init__(self, main_app):
        self.ui = main_app

        # [animal/race/image_name.jpg] = text
        self.image_text_mapping = load_json_file(self.ui.feature_folder + "/captions.json")

        # Will be initialized with the image features.
        # [image_name] = image_path (in imgDB)
        self.image_path_mapping = self._get_image_path_mapping()


        # Links for multimodal buttons
        self.ui.quitter_mult.clicked.connect(self.ui.Quitter)
        self.ui.charger_image_mult.clicked.connect(self.load_img_request)
        self.ui.charger_desc_mult.clicked.connect(self.load_features)
        self.ui.clear_mult.clicked.connect(self.reset_values)
        self.ui.chercher_mult.clicked.connect(self.search)

        # Attributes used in the class
        self.file_name = None
        self.text_request = None
        self.images_proches = []


        # Features indexes
        # [image_name] = feature_vector
        self.image_features = {}
        self.text_features = {}

        # Setup parameters for multimodal tab
        self.available_modes = {
            0: "Text",
            1: "Image",
            2: "Combined",
        }
        self.chosen_mode = -1

        self.unloaded_color = "#d60000"
        self.loaded_color = "#01aa29"

    def reset_values(self):
        """
        Reset the values of the attributes used in the class.
        """
        self.file_name = None
        self.text_request = None
        self.chosen_mode = -1

        self.ui.label_requete_mult.clear()
        self.ui.text_request.clear()

        self.image_features = {}
        self.text_features = {}
        self.ui.progressBar_mult.setValue(0)
        self.ui.charger_desc_mult.setStyleSheet(f"background-color: {self.unloaded_color};")

    def _get_image_path_mapping(self):
        mapping = {}
        for root, _, files in os.walk("imgDB"):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    mapping[os.path.splitext(f)[0]] = os.path.join(root, f)
        return mapping

    def load_img_request(self):
        self.file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpeg *.jpg *.bmp)"
        )

        if self.file_name:
            pixmap = QtGui.QPixmap(self.file_name)
            pixmap = pixmap.scaled(self.ui.label_requete_mult.width(),
                                   self.ui.label_requete_mult.height(),
                                   QtCore.Qt.KeepAspectRatio)
            self.ui.label_requete_mult.setPixmap(pixmap)
            self.ui.label_requete_mult.setAlignment(QtCore.Qt.AlignCenter)

        else:
            self.ui.show_error("Erreur !", "Aucune image sélectionnée. Veuillez sélectionner un image.")

    def _determine_mode(self):
        """
        Guess the mode based on the values loaded. (Text and/or Image)

        It updates the `self.chosen_mode` attribute based on the available modes.
        """
        self.text_request = self.ui.text_request.toPlainText().strip()

        if self.file_name and self.text_request:
            self.chosen_mode = 2
        elif self.file_name:
            self.chosen_mode = 1
        elif self.text_request:
            self.chosen_mode = 0
        else:
            self.chosen_mode = -1  # No valid mode
            self.ui.show_error("Erreur !", "Aucune image ou texte sélectionné. Veuillez sélectionner une image ou entrer un texte.")

    def load_features(self):
        pas = 0
        max_images = len(self.image_path_mapping)
        self.ui.progressBar_rech.setValue(0)
        for name in self.image_path_mapping.keys():
            # text features
            if len(self.image_features) < max_images:
                feature = np.loadtxt(f"{self.ui.feature_folder}/image_features/{name}.txt")
                self.image_features[name] = feature
            # text features
            if len(self.text_features) < max_images:
                feature = np.loadtxt(f"{self.ui.feature_folder}/text_features/{name}.txt")
                self.text_features[name] = feature
            pas += 1
            self.ui.progressBar_mult.setValue(int(100 * (pas / max_images)))

        self.ui.charger_desc_mult.setStyleSheet(f"background-color: {self.loaded_color};")

    def _text_search(self, k):
        # Load features '1_0_chiens_Siberianhusky_828.txt'
        feature = np.loadtxt(f"{self.ui.feature_folder}/text_features/1_0_chiens_Siberianhusky_828.txt")

        return get_k_neighbors(self.text_features, feature, k)

    def _image_search(self, k):
        # We don't have any GPU :(
        base_name = os.path.basename(self.file_name)
        # Remove the extension
        base_name = os.path.splitext(base_name)[0]

        req_features = self.image_features[base_name]

        return get_k_neighbors(self.image_features, req_features, k)

    def _combined_search(self, k):
        return []

    def _clear_scroll_area(self):
        layout = self.ui.scrollArea_content_mult.layout()
        # Remove layout from the scroll area
        if layout is not None:
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
            layout.deleteLater()

    def search(self):
        """
        Search the best matches based on the chosen mode.
        """
        if not self.text_features or not self.image_features:
            self.ui.show_error("Erreur !", "Aucun descripteur chargé. Veuillez charger les descripteurs d'images et de texte.")
            return

        self._determine_mode()
        if self.chosen_mode == -1:
            self.ui.show_error("Erreur !", "Aucune image ou texte sélectionné. Veuillez sélectionner une image ou entrer un texte.")
            return

        k = int(self.ui.comboBox_top_mult.currentText())
        neighbors = []
        if self.chosen_mode == 0:
            neighbors = self._text_search(k)
        elif self.chosen_mode == 1:
            neighbors = self._image_search(k)
        elif self.chosen_mode == 2:
            neighbors = self._combined_search(k)

        self._display_result(neighbors)

    def _display_result(self, neighbors):
        self._clear_scroll_area()

        # Configuration générale
        target_width = 300
        target_height = 200

        # Créer un layout vertical dynamiquement
        vbox_layout = QtWidgets.QVBoxLayout()
        vbox_layout.setAlignment(QtCore.Qt.AlignTop)

        # Lier le layout au widget de contenu scrollable

        # Ajout des blocs image + texte
        for index in range(len(neighbors)):
            image_name = neighbors[index][0]
            img = cv2.imread(self.image_path_mapping[image_name], 1)
            if img is None:
                continue

            # Traitement image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qImg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0],
                                QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            scaled_pixmap = pixmap.scaled(
                target_width,
                target_height,
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation
            )

            # QLabel pour image
            image_label = QtWidgets.QLabel()
            image_label.setPixmap(scaled_pixmap)
            image_label.setFixedSize(target_width, target_height)
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            image_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")

            # QLabel pour le texte
            caption = self.image_text_mapping.get(f"{image_name}.jpg", "No caption available")
            text_label = QtWidgets.QLabel(caption)
            text_label.setAlignment(QtCore.Qt.AlignCenter)
            text_label.setWordWrap(True)
            text_label.setStyleSheet("padding: 6px; font-size: 11pt;")

            # Layout vertical image + texte
            block_layout = QtWidgets.QVBoxLayout()
            block_layout.setContentsMargins(10, 10, 10, 10)
            block_layout.setSpacing(10)
            block_layout.addWidget(image_label)
            block_layout.addWidget(text_label)

            # Widget bloc
            block_widget = QtWidgets.QWidget()
            block_widget.setLayout(block_layout)
            block_widget.setMinimumHeight(target_height + 80)
            block_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            block_widget.setStyleSheet("""
                        background-color: #f9f9f9;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                    """)

            # Ajouter le bloc au layout principal
            vbox_layout.addWidget(block_widget)

        self.ui.scrollArea_content_mult.setLayout(vbox_layout)


