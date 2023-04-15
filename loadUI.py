from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit
from PyQt5 import uic
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore    import *
from PyQt5.QtGui     import *


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("scale_layout.ui", self)
        
        # find the widgets in the xml file

        # self.textedit = self.findChild(QTextEdit, "textEdit")
        # self.button = self.findChild(QPushButton, "pushButton")
        # self.button.clicked.connect(self.clickedBtn)
        QSizeGrip(self.size_grip)
        self.show()


app = QApplication(sys.argv)
window = UI()
app.exec_()