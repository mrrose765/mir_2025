import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from main_ui import Ui_MainWindow
from tab_index import TabIndexController
from tab_recherche import TabRechercheController




class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)


        self.tab_index = TabIndexController(self)
        self.tab_recherche = TabRechercheController(self)

    def Quitter(self):
        """ Fonction pour quitter l'application """
        print("Fermeture de l'application...")
        QtWidgets.QApplication.instance().quit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())