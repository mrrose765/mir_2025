import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox

from main_ui import (Ui_MainWindow)
from tab_index import TabIndexController
from tab_recherche import TabRechercheController
from tab_mult import TabMultimodalController




class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.features_folder = "features"
        self.image_folder = "imgDB"

        # Load controllers for each tab (Contains the logic)
        self.tab_index = TabIndexController(self)
        self.tab_recherche = TabRechercheController(self)
        self.tab_multimodal = TabMultimodalController(self)

    def Quitter(self):
        print("Fermeture de l'application...")
        QtWidgets.QApplication.instance().quit()

    @staticmethod
    def show_error(title, message):
        """
        Used to display an error message as pop-up.
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())