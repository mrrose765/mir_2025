# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'part1.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import os
import functions as f
import time
import numpy as np
import math
import cv2
import distances as d
import fnmatch
import matplotlib.pyplot as plt
from metrics import precision_at_k, recall, average_precision, r_precision, mean_average_precision

filenames = "imgDB"

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1209, 614)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1201, 571))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_index = QtWidgets.QWidget()
        self.tab_index.setObjectName("tab_index")
        self.checkBox_SIFT_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_SIFT_index.setGeometry(QtCore.QRect(20, 10, 85, 21))
        self.checkBox_SIFT_index.setObjectName("checkBox_SIFT_index")
        self.checkBox_ORB_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_ORB_index.setGeometry(QtCore.QRect(140, 10, 85, 21))
        self.checkBox_ORB_index.setObjectName("checkBox_ORB_index")
        self.checkBox_HSV_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_HSV_index.setGeometry(QtCore.QRect(470, 10, 101, 21))
        self.checkBox_HSV_index.setObjectName("checkBox_HSV_index")
        self.checkBox_HistC_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_HistC_index.setGeometry(QtCore.QRect(270, 10, 121, 21))
        self.checkBox_HistC_index.setObjectName("checkBox_HistC_index")
        self.checkBox_GLCM_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_GLCM_index.setGeometry(QtCore.QRect(650, 10, 81, 21))
        self.checkBox_GLCM_index.setObjectName("checkBox_GLCM_index")
        self.checkBox_LBP_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_LBP_index.setGeometry(QtCore.QRect(800, 10, 61, 21))
        self.checkBox_LBP_index.setObjectName("checkBox_LBP_index")
        self.checkBox_HOG_index = QtWidgets.QCheckBox(self.tab_index)
        self.checkBox_HOG_index.setGeometry(QtCore.QRect(930, 10, 71, 21))
        self.checkBox_HOG_index.setObjectName("checkBox_HOG_index")
        self.charger = QtWidgets.QPushButton(self.tab_index)
        self.charger.setGeometry(QtCore.QRect(10, 50, 441, 51))
        self.charger.setObjectName("charger")
        self.quitter = QtWidgets.QPushButton(self.tab_index)
        self.quitter.setGeometry(QtCore.QRect(930, 50, 71, 51))
        self.quitter.setObjectName("quitter")
        self.indexer = QtWidgets.QPushButton(self.tab_index)
        self.indexer.setGeometry(QtCore.QRect(470, 50, 451, 51))
        self.indexer.setObjectName("indexer")
        self.imagelabel = QtWidgets.QLabel(self.tab_index)
        self.imagelabel.setGeometry(QtCore.QRect(10, 110, 261, 231))
        self.imagelabel.setAutoFillBackground(False)
        self.imagelabel.setText("")
        self.imagelabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imagelabel.setObjectName("imagelabel")
        self.image = QtWidgets.QLabel(self.tab_index)
        self.image.setGeometry(QtCore.QRect(10, 160, 441, 331))
        self.image.setFrameShape(QtWidgets.QFrame.Panel)
        self.image.setText("")
        self.image.setScaledContents(True)
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setObjectName("image")
        self.label_3 = QtWidgets.QLabel(self.tab_index)
        self.label_3.setGeometry(QtCore.QRect(470, 120, 531, 31))
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.tab_index)
        self.label.setGeometry(QtCore.QRect(10, 120, 441, 31))
        self.label.setFrameShape(QtWidgets.QFrame.Panel)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.progressBar = QtWidgets.QProgressBar(self.tab_index)
        self.progressBar.setGeometry(QtCore.QRect(10, 510, 1011, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.tableView = QtWidgets.QTableView(self.tab_index)
        self.tableView.setGeometry(QtCore.QRect(470, 160, 531, 331))
        self.tableView.setObjectName("tableView")
        self.tabWidget.addTab(self.tab_index, "")
        self.tab_rech = QtWidgets.QWidget()
        self.tab_rech.setObjectName("tab_rech")
        self.label_2 = QtWidgets.QLabel(self.tab_rech)
        self.label_2.setGeometry(QtCore.QRect(100, 10, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.checkBox_HistC_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_HistC_rech.setGeometry(QtCore.QRect(100, 50, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.checkBox_HistC_rech.setFont(font)
        self.checkBox_HistC_rech.setObjectName("checkBox_HistC_rech")
        self.checkBox_HSV_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_HSV_rech.setGeometry(QtCore.QRect(170, 50, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_HSV_rech.setFont(font)
        self.checkBox_HSV_rech.setObjectName("checkBox_HSV_rech")
        self.checkBox_SIFT_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_SIFT_rech.setGeometry(QtCore.QRect(100, 80, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_SIFT_rech.setFont(font)
        self.checkBox_SIFT_rech.setObjectName("checkBox_SIFT_rech")
        self.checkBox_ORB_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_ORB_rech.setGeometry(QtCore.QRect(170, 80, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_ORB_rech.setFont(font)
        self.checkBox_ORB_rech.setObjectName("checkBox_ORB_rech")
        self.checkBox_GLCM_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_GLCM_rech.setGeometry(QtCore.QRect(170, 110, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_GLCM_rech.setFont(font)
        self.checkBox_GLCM_rech.setObjectName("checkBox_GLCM_rech")
        self.checkBox_LBP_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_LBP_rech.setGeometry(QtCore.QRect(100, 110, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_LBP_rech.setFont(font)
        self.checkBox_LBP_rech.setObjectName("checkBox_LBP_rech")
        self.checkBox_HOG_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_HOG_rech.setGeometry(QtCore.QRect(250, 50, 71, 17))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_HOG_rech.setFont(font)
        self.checkBox_HOG_rech.setObjectName("checkBox_HOG_rech")
        self.checkBox_Moments_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_Moments_rech.setGeometry(QtCore.QRect(250, 80, 81, 17))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_Moments_rech.setFont(font)
        self.checkBox_Moments_rech.setObjectName("checkBox_Moments_rech")
        self.label_4 = QtWidgets.QLabel(self.tab_rech)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_requete = QtWidgets.QLabel(self.tab_rech)
        self.label_requete.setGeometry(QtCore.QRect(10, 190, 311, 291))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_requete.setFont(font)
        self.label_requete.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_requete.setText("")
        self.label_requete.setScaledContents(True)
        self.label_requete.setAlignment(QtCore.Qt.AlignCenter)
        self.label_requete.setObjectName("label_requete")
        self.progressBar_rech = QtWidgets.QProgressBar(self.tab_rech)
        self.progressBar_rech.setGeometry(QtCore.QRect(10, 490, 921, 41))
        self.progressBar_rech.setProperty("value", 0)
        self.progressBar_rech.setObjectName("progressBar_rech")
        self.label_5 = QtWidgets.QLabel(self.tab_rech)
        self.label_5.setGeometry(QtCore.QRect(330, 10, 481, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_rech)
        self.label_6.setGeometry(QtCore.QRect(820, 10, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.quitter_rech = QtWidgets.QPushButton(self.tab_rech)
        self.quitter_rech.setGeometry(QtCore.QRect(940, 490, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.quitter_rech.setFont(font)
        self.quitter_rech.setObjectName("quitter_rech")
        self.comboBox = QtWidgets.QComboBox(self.tab_rech)
        self.comboBox.setGeometry(QtCore.QRect(460, 100, 221, 41))
        self.comboBox.setObjectName("comboBox")
        self.label_7 = QtWidgets.QLabel(self.tab_rech)
        self.label_7.setGeometry(QtCore.QRect(330, 100, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_rech)
        self.label_8.setGeometry(QtCore.QRect(330, 150, 481, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab_rech)
        self.label_9.setGeometry(QtCore.QRect(820, 270, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.chercher = QtWidgets.QPushButton(self.tab_rech)
        self.chercher.setGeometry(QtCore.QRect(690, 100, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.chercher.setFont(font)
        self.chercher.setObjectName("chercher")
        self.calcul_RP = QtWidgets.QPushButton(self.tab_rech)
        self.calcul_RP.setGeometry(QtCore.QRect(1000, 50, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.calcul_RP.setFont(font)
        self.calcul_RP.setObjectName("calcul_RP")
        self.checkBox_autre_rech = QtWidgets.QCheckBox(self.tab_rech)
        self.checkBox_autre_rech.setGeometry(QtCore.QRect(250, 110, 81, 17))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_autre_rech.setFont(font)
        self.checkBox_autre_rech.setObjectName("checkBox_autre_rech")
        self.charger_rech = QtWidgets.QPushButton(self.tab_rech)
        self.charger_rech.setGeometry(QtCore.QRect(10, 60, 81, 61))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.charger_rech.setFont(font)
        self.charger_rech.setObjectName("charger_rech")
        self.label_10 = QtWidgets.QLabel(self.tab_rech)
        self.label_10.setGeometry(QtCore.QRect(10, 10, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.charger_desc = QtWidgets.QPushButton(self.tab_rech)
        self.charger_desc.setGeometry(QtCore.QRect(330, 50, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.charger_desc.setFont(font)
        self.charger_desc.setObjectName("charger_desc")
        self.label_courbe = QtWidgets.QLabel(self.tab_rech)
        self.label_courbe.setGeometry(QtCore.QRect(820, 310, 371, 171))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_courbe.setFont(font)
        self.label_courbe.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_courbe.setText("")
        self.label_courbe.setScaledContents(True)
        self.label_courbe.setAlignment(QtCore.Qt.AlignCenter)
        self.label_courbe.setObjectName("label_courbe")
        self.pushButton = QtWidgets.QPushButton(self.tab_rech)
        self.pushButton.setGeometry(QtCore.QRect(820, 50, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_11 = QtWidgets.QLabel(self.tab_rech)
        self.label_11.setGeometry(QtCore.QRect(820, 100, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_metriques = QtWidgets.QLabel(self.tab_rech)
        self.label_metriques.setGeometry(QtCore.QRect(820, 140, 371, 121))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_metriques.setFont(font)
        self.label_metriques.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_metriques.setText("")
        self.label_metriques.setScaledContents(True)
        self.label_metriques.setAlignment(QtCore.Qt.AlignCenter)
        self.label_metriques.setObjectName("label_metriques")
        self.label_12 = QtWidgets.QLabel(self.tab_rech)
        self.label_12.setGeometry(QtCore.QRect(540, 50, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.comboBox_top = QtWidgets.QComboBox(self.tab_rech)
        self.comboBox_top.setGeometry(QtCore.QRect(630, 50, 181, 41))
        self.comboBox_top.setObjectName("comboBox_top")
        self.scrollArea = QtWidgets.QScrollArea(self.tab_rech)
        self.scrollArea.setGeometry(QtCore.QRect(330, 190, 481, 291))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tabWidget.addTab(self.tab_rech, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1209, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox_SIFT_index.setText(_translate("MainWindow", "SIFT"))
        self.checkBox_ORB_index.setText(_translate("MainWindow", "ORB"))
        self.checkBox_HSV_index.setText(_translate("MainWindow", "Hist HSV"))
        self.checkBox_HistC_index.setText(_translate("MainWindow", "Hist Couleur"))
        self.checkBox_GLCM_index.setText(_translate("MainWindow", "GLCM"))
        self.checkBox_LBP_index.setText(_translate("MainWindow", "LBP"))
        self.checkBox_HOG_index.setText(_translate("MainWindow", "HOG"))
        self.charger.setText(_translate("MainWindow", "Charger et afficher la base de données"))
        self.quitter.setText(_translate("MainWindow", "Quitter"))
        self.indexer.setText(_translate("MainWindow", "Calculer les descripteurs"))
        self.label_3.setText(_translate("MainWindow", "Base d\'images"))
        self.label.setText(_translate("MainWindow", "Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_index), _translate("MainWindow", "Indexation"))
        self.label_2.setText(_translate("MainWindow", "Choix de descripteur"))
        self.checkBox_HistC_rech.setText(_translate("MainWindow", "BGR"))
        self.checkBox_HSV_rech.setText(_translate("MainWindow", "HSV"))
        self.checkBox_SIFT_rech.setText(_translate("MainWindow", "SIFT"))
        self.checkBox_ORB_rech.setText(_translate("MainWindow", "ORB"))
        self.checkBox_GLCM_rech.setText(_translate("MainWindow", "GLCM"))
        self.checkBox_LBP_rech.setText(_translate("MainWindow", "LBP"))
        self.checkBox_HOG_rech.setText(_translate("MainWindow", "HOG"))
        self.checkBox_Moments_rech.setText(_translate("MainWindow", "Mom."))
        self.label_4.setText(_translate("MainWindow", "Image requête"))
        self.label_5.setText(_translate("MainWindow", "Recherche"))
        self.label_6.setText(_translate("MainWindow", "Rappel/Précision"))
        self.quitter_rech.setText(_translate("MainWindow", "Quitter"))
        self.label_7.setText(_translate("MainWindow", "Distance :"))
        self.label_8.setText(_translate("MainWindow", "Résultats"))
        self.label_9.setText(_translate("MainWindow", "Courbe R/P"))
        self.chercher.setText(_translate("MainWindow", "Recherche"))
        self.calcul_RP.setText(_translate("MainWindow", "Calculer la courbe R/P"))
        self.checkBox_autre_rech.setText(_translate("MainWindow", "Autre"))
        self.charger_rech.setText(_translate("MainWindow", "Charger"))
        self.label_10.setText(_translate("MainWindow", "Requête"))
        self.charger_desc.setText(_translate("MainWindow", "Charger descripteurs"))
        self.pushButton.setText(_translate("MainWindow", "Calculer métriques"))
        self.label_11.setText(_translate("MainWindow", "Métriques"))
        self.label_12.setText(_translate("MainWindow", "Top :"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_rech), _translate("MainWindow", "Recherche"))

        # Liens des boutons pour indexation
        self.charger.clicked.connect(self.Ouvrir_index)
        self.quitter.clicked.connect(self.Quitter)
        self.tableView.clicked.connect(self.CliquerTab)
        self.indexer.clicked.connect(self.extractFeatures)

        #Liens des boutons pour recherche
        self.comboBox_top.addItems(["20", "50"])
        self.quitter_rech.clicked.connect(self.Quitter)
        self.charger_rech.clicked.connect(self.OuvrirImage)
        self.charger_desc.clicked.connect(self.loadFeatures)
        self.chercher.clicked.connect(self.Recherche)
        self.calcul_RP.clicked.connect(self.rappel_precision)
        self.pushButton.clicked.connect(self.afficher_metriques)

    def Ouvrir_index(self, MainWindow):
        """Fonction pour ouvrir un explorateur de fichiers et charger les images"""
        self.list_images_index = []
        self.Dossier_images_index = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select directory', "C://", QtWidgets.QFileDialog.ShowDirsOnly)+"/"

        extensions_images = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')
        for root, dirs, files in os.walk(self.Dossier_images_index):
            for file in files:
                if file.lower().endswith(extensions_images):
                    chemin_complet = os.path.join(root, file)
                    self.list_images_index.append(chemin_complet)

        # Affichage de la première image
        if self.list_images_index:
            pixmap = QtGui.QPixmap(self.list_images_index[0])
            pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio)
            self.image.setPixmap(pixmap)
            self.image.setAlignment(QtCore.Qt.AlignCenter)

        # Remplir le tableView avec les noms de fichiers
        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(["File name"])

        for chemin_image in self.list_images_index:
            nom_fichier = os.path.basename(chemin_image)
            item = QtGui.QStandardItem(nom_fichier)  
            item.setEditable(False)
            item.setData(chemin_image, QtCore.Qt.UserRole)
            model.appendRow([item])

        self.tableView.setModel(model)

    def Quitter(self):
        """ Fonction pour quitter l'application """
        print("Fermeture de l'application...")
        QtWidgets.QApplication.instance().quit()

    def CliquerTab(self, MainWindow):
        index = self.tableView.selectionModel().currentIndex()
        chemin_complet = index.data(QtCore.Qt.UserRole)  # Récupère le chemin complet
        if chemin_complet and os.path.exists(chemin_complet):
            pixmap = QtGui.QPixmap(chemin_complet)
            pixmap = pixmap.scaled(self.image.width(), self.image.height(), QtCore.Qt.KeepAspectRatio)
            self.image.setPixmap(pixmap)
            self.image.setAlignment(QtCore.Qt.AlignCenter)

    def extractFeatures(self, MainWindow):
        # Vérifie que des images sont bien chargées
        if not hasattr(self, 'list_images_index') or len(self.list_images_index) < 1:
            print("Merci de charger la base de données avec le bouton Ouvrir")
            return

        # Vérifie que l'utilisateur a coché au moins un descripteur
        if not (self.checkBox_HistC_index.isChecked() or self.checkBox_HSV_index.isChecked() or 
                self.checkBox_SIFT_index.isChecked() or self.checkBox_ORB_index.isChecked() or 
                self.checkBox_GLCM_index.isChecked() or self.checkBox_LBP_index.isChecked() or
                self.checkBox_HOG_index.isChecked()):
            print("Merci de sélectionner un descripteur via le menu ...")
            f.showDialog()
            return

        # Exécute les descripteurs cochés
        if self.checkBox_HistC_index.isChecked():
            f.generateHistogramme_Color(self.list_images_index, self.progressBar)

        if self.checkBox_HSV_index.isChecked():
            f.generateHistogramme_HSV(self.list_images_index, self.progressBar)

        if self.checkBox_SIFT_index.isChecked():
            f.generateSIFT(self.list_images_index, self.progressBar)

        if self.checkBox_ORB_index.isChecked():
            f.generateORB(self.list_images_index, self.progressBar)

        if self.checkBox_GLCM_index.isChecked():
            f.generateGLCM(self.list_images_index, self.progressBar)

        if self.checkBox_LBP_index.isChecked():
            f.generateLBP(self.list_images_index, self.progressBar)

        if self.checkBox_HOG_index.isChecked():
            f.generateHOG(self.list_images_index, self.progressBar)

        print("Indexation terminée.")

    def OuvrirImage(self, MainWindow):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpeg *.jpg *.bmp)"
        )
        if self.fileName:
            pixmap = QtGui.QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.label_requete.width(),
                                self.label_requete.height(), QtCore.Qt.KeepAspectRatio)
            self.label_requete.setPixmap(pixmap)
            self.label_requete.setAlignment(QtCore.Qt.AlignCenter)
        else:
            print("Aucune image requête sélectionnée. Dans ouvrir")

    def loadFeatures(self, MainWindow):

        self.algo_choice = 0
        folder_model = ""

        # Sélection du dossier selon le descripteur
        if self.checkBox_HistC_rech.isChecked():
            folder_model = './BGR'
            self.algo_choice = 1
        elif self.checkBox_HSV_rech.isChecked():
            folder_model = './HSV'
            self.algo_choice = 2
        elif self.checkBox_SIFT_rech.isChecked():
            folder_model = './SIFT'
            self.algo_choice = 3
        elif self.checkBox_ORB_rech.isChecked():
            folder_model = './ORB'
            self.algo_choice = 4
        elif self.checkBox_GLCM_rech.isChecked():
            folder_model = './GLCM'
            self.algo_choice = 5
        elif self.checkBox_LBP_rech.isChecked():
            folder_model = './LBP'
            self.algo_choice = 6
        elif self.checkBox_HOG_rech.isChecked():
            folder_model = './HOG'
            self.algo_choice = 7
        elif self.checkBox_Moments_rech.isChecked():
            folder_model = './MOMENTS'
            self.algo_choice = 8
        else:
            print("Merci de sélectionner un descripteur")
            self.showDialog()
            return

        # Nettoyer la grille d'affichage
        for i in reversed(range(self.gridLayout.count())):
            self.gridLayout.itemAt(i).widget().setParent(None)

        # Mise à jour des distances
        if self.algo_choice in [3, 4, 5, 6, 7, 8]:
            self.comboBox.clear()
            self.comboBox.addItems(["Brute force", "Flann"])
        else:
            self.comboBox.clear()
            self.comboBox.addItems(["Euclidienne", "Correlation", "Chi carre", "Intersection", "Bhattacharyya"])

        # Vérification image requête
        if len(filenames) < 1:
            print("Merci de charger une image avec le bouton Ouvrir")
            return

        # Chargement
        print("Chargement descripteurs en cours ...")
        start_time = time.time()
        self.features1 = []
        pas = 0

        # Construire un index de toutes les images dans imgDB
        image_index = {}
        for root, _, files in os.walk("imgDB"):
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
            self.progressBar_rech.setValue(int(100 * (pas / total_files)))

        end_time = time.time()
        print(f"Temps de chargement du descripteur {folder_model} : {end_time - start_time:.4f} secondes")


    def Recherche(self, MainWindow):
        for i in reversed(range(self.gridLayout.count())):
            self.gridLayout.itemAt(i).widget().setParent(None)

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
        self.sortie = int(self.comboBox_top.currentText())
        distanceName = self.comboBox.currentText()

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

                self.gridLayout.addWidget(label, i, j)

        end_time = time.time()
        print(f"Temps de recherche pour le descripteur {self.algo_choice} : {end_time - start_time:.4f} secondes")

    def rappel_precision(self):
        start_time = time.time()

        rappel_precision=[]
        rappels=[]
        precisions=[]

        filename_req=os.path.basename(self.fileName)
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

        #Création de la courbe R/P
        plt.plot(rappels, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("R/P"+str(self.sortie)+" voisins de l'image")

        #Enregistrement de la courbe RP
        save_folder=os.path.join(".","rp_courbes")
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

        label_width = self.label_requete.frameGeometry().width()
        label_height = self.label_requete.frameGeometry().height()
        self.label_courbe.setAlignment(QtCore.Qt.AlignCenter)
        self.label_courbe.setPixmap(pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        end_time = time.time()
        print(f"Temps de calcul de la courbe R/P: {end_time - start_time:.4f} secondes")

    def afficher_metriques(self):
        if not hasattr(self, "nom_image_plus_proches") or not self.nom_image_plus_proches:
            self.label_metriques.setText("Aucun résultat de recherche.")
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
        
        self.label_metriques.setText(texte)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
