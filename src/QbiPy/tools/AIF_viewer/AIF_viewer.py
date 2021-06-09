# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AIF_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AIFViewer(object):
    def setupUi(self, AIFViewer):
        AIFViewer.setObjectName("AIFViewer")
        AIFViewer.resize(1772, 937)
        self.centralwidget = QtWidgets.QWidget(AIFViewer)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.dynVolLabel = QtWidgets.QLabel(self.frame)
        self.dynVolLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.dynVolLabel.setObjectName("dynVolLabel")
        self.gridLayout_2.addWidget(self.dynVolLabel, 0, 1, 1, 2)
        self.colorbarGraphicsView = QtWidgets.QGraphicsView(self.frame)
        self.colorbarGraphicsView.setMaximumSize(QtCore.QSize(800, 20))
        self.colorbarGraphicsView.setObjectName("colorbarGraphicsView")
        self.gridLayout_2.addWidget(self.colorbarGraphicsView, 2, 1, 1, 1)
        self.minContrast1 = QtWidgets.QLabel(self.frame)
        self.minContrast1.setMaximumSize(QtCore.QSize(30, 16777215))
        self.minContrast1.setObjectName("minContrast1")
        self.gridLayout_2.addWidget(self.minContrast1, 2, 0, 1, 1)
        self.leftGraphicsView = QtWidgets.QGraphicsView(self.frame)
        self.leftGraphicsView.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.leftGraphicsView.setObjectName("leftGraphicsView")
        self.gridLayout_2.addWidget(self.leftGraphicsView, 1, 0, 1, 3)
        self.maxContrast1 = QtWidgets.QLabel(self.frame)
        self.maxContrast1.setMaximumSize(QtCore.QSize(30, 16777215))
        self.maxContrast1.setObjectName("maxContrast1")
        self.gridLayout_2.addWidget(self.maxContrast1, 2, 2, 1, 1)
        self.aifPlotWidget = MplWidget(self.frame)
        self.aifPlotWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.aifPlotWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.aifPlotWidget.setObjectName("aifPlotWidget")
        self.gridLayout_2.addWidget(self.aifPlotWidget, 0, 3, 2, 2)
        self.horizontalLayout.addWidget(self.frame)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.previousAifButton = QtWidgets.QPushButton(self.groupBox)
        self.previousAifButton.setObjectName("previousAifButton")
        self.gridLayout_3.addWidget(self.previousAifButton, 5, 0, 1, 1)
        self.nextAifButton = QtWidgets.QPushButton(self.groupBox)
        self.nextAifButton.setObjectName("nextAifButton")
        self.gridLayout_3.addWidget(self.nextAifButton, 5, 3, 1, 1)
        self.selectAifLabel = QtWidgets.QLabel(self.groupBox)
        self.selectAifLabel.setObjectName("selectAifLabel")
        self.gridLayout_3.addWidget(self.selectAifLabel, 4, 0, 1, 2)
        self.aifDirLineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.aifDirLineEdit.setObjectName("aifDirLineEdit")
        self.gridLayout_3.addWidget(self.aifDirLineEdit, 3, 0, 1, 3)
        self.aifDirSelectButton = QtWidgets.QPushButton(self.groupBox)
        self.aifDirSelectButton.setObjectName("aifDirSelectButton")
        self.gridLayout_3.addWidget(self.aifDirSelectButton, 3, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 2, 0, 1, 2)
        self.aifComboBox = QtWidgets.QComboBox(self.groupBox)
        self.aifComboBox.setObjectName("aifComboBox")
        self.gridLayout_3.addWidget(self.aifComboBox, 5, 1, 1, 2)
        self.aifInfoTextEdit = QtWidgets.QTextEdit(self.groupBox)
        self.aifInfoTextEdit.setObjectName("aifInfoTextEdit")
        self.gridLayout_3.addWidget(self.aifInfoTextEdit, 8, 0, 1, 4)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 9, 1, 1, 1)
        self.frame_8 = QtWidgets.QFrame(self.groupBox)
        self.frame_8.setMinimumSize(QtCore.QSize(0, 100))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout.setObjectName("verticalLayout")
        self.selectSliceLabel = QtWidgets.QLabel(self.frame_8)
        self.selectSliceLabel.setObjectName("selectSliceLabel")
        self.verticalLayout.addWidget(self.selectSliceLabel)
        self.sliceSlider = QtWidgets.QSlider(self.frame_8)
        self.sliceSlider.setOrientation(QtCore.Qt.Horizontal)
        self.sliceSlider.setObjectName("sliceSlider")
        self.verticalLayout.addWidget(self.sliceSlider)
        self.gridLayout_3.addWidget(self.frame_8, 6, 0, 1, 4)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout.setObjectName("gridLayout")
        self.maxContrast = QtWidgets.QLabel(self.groupBox_8)
        self.maxContrast.setObjectName("maxContrast")
        self.gridLayout.addWidget(self.maxContrast, 4, 0, 1, 1)
        self.contrast1Label = QtWidgets.QLabel(self.groupBox_8)
        self.contrast1Label.setObjectName("contrast1Label")
        self.gridLayout.addWidget(self.contrast1Label, 2, 0, 1, 5)
        self.minContrast = QtWidgets.QLabel(self.groupBox_8)
        self.minContrast.setObjectName("minContrast")
        self.gridLayout.addWidget(self.minContrast, 3, 0, 1, 1)
        self.frame_9 = QtWidgets.QFrame(self.groupBox_8)
        self.frame_9.setMinimumSize(QtCore.QSize(0, 40))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_9)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.showMarkersButton = QtWidgets.QPushButton(self.frame_9)
        self.showMarkersButton.setObjectName("showMarkersButton")
        self.horizontalLayout_2.addWidget(self.showMarkersButton)
        self.gridLayout.addWidget(self.frame_9, 5, 0, 1, 5)
        self.dynVolSelectButton = QtWidgets.QPushButton(self.groupBox_8)
        self.dynVolSelectButton.setObjectName("dynVolSelectButton")
        self.gridLayout.addWidget(self.dynVolSelectButton, 1, 4, 1, 1)
        self.minContrastSlider = QtWidgets.QSlider(self.groupBox_8)
        self.minContrastSlider.setMinimumSize(QtCore.QSize(150, 0))
        self.minContrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.minContrastSlider.setObjectName("minContrastSlider")
        self.gridLayout.addWidget(self.minContrastSlider, 3, 1, 1, 4)
        self.dynVolLineEdit = QtWidgets.QLineEdit(self.groupBox_8)
        self.dynVolLineEdit.setObjectName("dynVolLineEdit")
        self.gridLayout.addWidget(self.dynVolLineEdit, 1, 0, 1, 4)
        self.maxContrastSlider = QtWidgets.QSlider(self.groupBox_8)
        self.maxContrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.maxContrastSlider.setObjectName("maxContrastSlider")
        self.gridLayout.addWidget(self.maxContrastSlider, 4, 1, 1, 4)
        self.label_2 = QtWidgets.QLabel(self.groupBox_8)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 2)
        self.gridLayout_3.addWidget(self.groupBox_8, 7, 0, 1, 4)
        self.horizontalLayout.addWidget(self.groupBox)
        AIFViewer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AIFViewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1772, 21))
        self.menubar.setObjectName("menubar")
        AIFViewer.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AIFViewer)
        self.statusbar.setObjectName("statusbar")
        AIFViewer.setStatusBar(self.statusbar)

        self.retranslateUi(AIFViewer)
        QtCore.QMetaObject.connectSlotsByName(AIFViewer)

    def retranslateUi(self, AIFViewer):
        _translate = QtCore.QCoreApplication.translate
        AIFViewer.setWindowTitle(_translate("AIFViewer", "AIF viewer"))
        self.dynVolLabel.setText(_translate("AIFViewer", "Volume 1"))
        self.minContrast1.setText(_translate("AIFViewer", "Min"))
        self.maxContrast1.setText(_translate("AIFViewer", "Max"))
        self.groupBox.setTitle(_translate("AIFViewer", "Controls"))
        self.previousAifButton.setText(_translate("AIFViewer", "Previous"))
        self.nextAifButton.setText(_translate("AIFViewer", "Next"))
        self.selectAifLabel.setText(_translate("AIFViewer", "Select AIF:"))
        self.aifDirSelectButton.setText(_translate("AIFViewer", "Select"))
        self.label.setText(_translate("AIFViewer", "AIF folder:"))
        self.selectSliceLabel.setText(_translate("AIFViewer", "Select slice:"))
        self.maxContrast.setText(_translate("AIFViewer", "Max"))
        self.contrast1Label.setText(_translate("AIFViewer", "Min/max contrast"))
        self.minContrast.setText(_translate("AIFViewer", "Min"))
        self.showMarkersButton.setText(_translate("AIFViewer", "Show candidates"))
        self.dynVolSelectButton.setText(_translate("AIFViewer", "Select"))
        self.label_2.setText(_translate("AIFViewer", "Dynamic volume:"))
from QbiPy.tools.mplwidget import MplWidget
