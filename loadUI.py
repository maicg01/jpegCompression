from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit
from PyQt5 import uic
import sys


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("demo.ui", self)
        
        # find the widgets in the xml file

        # self.textedit = self.findChild(QTextEdit, "textEdit")
        # self.button = self.findChild(QPushButton, "pushButton")
        # self.button.clicked.connect(self.clickedBtn)

        self.show()


app = QApplication(sys.argv)
window = UI()
app.exec_()