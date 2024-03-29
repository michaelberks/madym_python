# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DCE_fit_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DCEFitViewer(object):
    def setupUi(self, DCEFitViewer):
        DCEFitViewer.setObjectName("DCEFitViewer")
        DCEFitViewer.resize(1772, 937)
        self.centralwidget = QtWidgets.QWidget(DCEFitViewer)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.dceDir = QtWidgets.QLabel(self.frame)
        self.dceDir.setMaximumSize(QtCore.QSize(16777215, 30))
        self.dceDir.setAlignment(QtCore.Qt.AlignCenter)
        self.dceDir.setObjectName("dceDir")
        self.verticalLayout.addWidget(self.dceDir)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.dcePlotWidget_11 = MplWidget(self.frame_2)
        self.dcePlotWidget_11.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_11.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_11.setObjectName("dcePlotWidget_11")
        self.gridLayout_2.addWidget(self.dcePlotWidget_11, 2, 0, 1, 1)
        self.dcePlotWidget_1 = MplWidget(self.frame_2)
        self.dcePlotWidget_1.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_1.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_1.setObjectName("dcePlotWidget_1")
        self.gridLayout_2.addWidget(self.dcePlotWidget_1, 0, 0, 1, 1)
        self.dcePlotWidget_6 = MplWidget(self.frame_2)
        self.dcePlotWidget_6.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_6.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_6.setObjectName("dcePlotWidget_6")
        self.gridLayout_2.addWidget(self.dcePlotWidget_6, 1, 0, 1, 1)
        self.dcePlotWidget_2 = MplWidget(self.frame_2)
        self.dcePlotWidget_2.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_2.setObjectName("dcePlotWidget_2")
        self.gridLayout_2.addWidget(self.dcePlotWidget_2, 0, 2, 1, 1)
        self.dcePlotWidget_8 = MplWidget(self.frame_2)
        self.dcePlotWidget_8.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_8.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_8.setObjectName("dcePlotWidget_8")
        self.gridLayout_2.addWidget(self.dcePlotWidget_8, 1, 3, 1, 1)
        self.dcePlotWidget_10 = MplWidget(self.frame_2)
        self.dcePlotWidget_10.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_10.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_10.setObjectName("dcePlotWidget_10")
        self.gridLayout_2.addWidget(self.dcePlotWidget_10, 1, 5, 1, 1)
        self.dcePlotWidget_15 = MplWidget(self.frame_2)
        self.dcePlotWidget_15.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_15.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_15.setObjectName("dcePlotWidget_15")
        self.gridLayout_2.addWidget(self.dcePlotWidget_15, 2, 5, 1, 1)
        self.dcePlotWidget_5 = MplWidget(self.frame_2)
        self.dcePlotWidget_5.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_5.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_5.setObjectName("dcePlotWidget_5")
        self.gridLayout_2.addWidget(self.dcePlotWidget_5, 0, 5, 1, 1)
        self.dcePlotWidget_7 = MplWidget(self.frame_2)
        self.dcePlotWidget_7.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_7.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_7.setObjectName("dcePlotWidget_7")
        self.gridLayout_2.addWidget(self.dcePlotWidget_7, 1, 2, 1, 1)
        self.dcePlotWidget_12 = MplWidget(self.frame_2)
        self.dcePlotWidget_12.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_12.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_12.setObjectName("dcePlotWidget_12")
        self.gridLayout_2.addWidget(self.dcePlotWidget_12, 2, 2, 1, 1)
        self.dcePlotWidget_13 = MplWidget(self.frame_2)
        self.dcePlotWidget_13.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_13.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_13.setObjectName("dcePlotWidget_13")
        self.gridLayout_2.addWidget(self.dcePlotWidget_13, 2, 3, 1, 1)
        self.dcePlotWidget_14 = MplWidget(self.frame_2)
        self.dcePlotWidget_14.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_14.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_14.setObjectName("dcePlotWidget_14")
        self.gridLayout_2.addWidget(self.dcePlotWidget_14, 2, 4, 1, 1)
        self.dcePlotWidget_9 = MplWidget(self.frame_2)
        self.dcePlotWidget_9.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_9.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_9.setObjectName("dcePlotWidget_9")
        self.gridLayout_2.addWidget(self.dcePlotWidget_9, 1, 4, 1, 1)
        self.dcePlotWidget_4 = MplWidget(self.frame_2)
        self.dcePlotWidget_4.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_4.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_4.setObjectName("dcePlotWidget_4")
        self.gridLayout_2.addWidget(self.dcePlotWidget_4, 0, 4, 1, 1)
        self.dcePlotWidget_3 = MplWidget(self.frame_2)
        self.dcePlotWidget_3.setMinimumSize(QtCore.QSize(0, 0))
        self.dcePlotWidget_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.dcePlotWidget_3.setObjectName("dcePlotWidget_3")
        self.gridLayout_2.addWidget(self.dcePlotWidget_3, 0, 3, 1, 1)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.dceDirLineEdit = QtWidgets.QLineEdit(self.frame_3)
        self.dceDirLineEdit.setMinimumSize(QtCore.QSize(0, 0))
        self.dceDirLineEdit.setObjectName("dceDirLineEdit")
        self.horizontalLayout_2.addWidget(self.dceDirLineEdit)
        self.dceDirSelectButton = QtWidgets.QPushButton(self.frame_3)
        self.dceDirSelectButton.setObjectName("dceDirSelectButton")
        self.horizontalLayout_2.addWidget(self.dceDirSelectButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.voxelsLabel = QtWidgets.QLabel(self.frame_3)
        self.voxelsLabel.setObjectName("voxelsLabel")
        self.horizontalLayout_2.addWidget(self.voxelsLabel)
        self.previousButton = QtWidgets.QPushButton(self.frame_3)
        self.previousButton.setObjectName("previousButton")
        self.horizontalLayout_2.addWidget(self.previousButton)
        self.voxelComboBox = QtWidgets.QComboBox(self.frame_3)
        self.voxelComboBox.setObjectName("voxelComboBox")
        self.horizontalLayout_2.addWidget(self.voxelComboBox)
        self.nextButton = QtWidgets.QPushButton(self.frame_3)
        self.nextButton.setObjectName("nextButton")
        self.horizontalLayout_2.addWidget(self.nextButton)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.minCtSpinBox = QtWidgets.QDoubleSpinBox(self.frame_3)
        self.minCtSpinBox.setDecimals(3)
        self.minCtSpinBox.setMinimum(-99.9)
        self.minCtSpinBox.setSingleStep(0.1)
        self.minCtSpinBox.setObjectName("minCtSpinBox")
        self.horizontalLayout_2.addWidget(self.minCtSpinBox)
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.maxCtSpinBox = QtWidgets.QDoubleSpinBox(self.frame_3)
        self.maxCtSpinBox.setDecimals(3)
        self.maxCtSpinBox.setSingleStep(0.1)
        self.maxCtSpinBox.setObjectName("maxCtSpinBox")
        self.horizontalLayout_2.addWidget(self.maxCtSpinBox)
        spacerItem3 = QtWidgets.QSpacerItem(206, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.verticalLayout.addWidget(self.frame_3)
        self.horizontalLayout.addWidget(self.frame)
        DCEFitViewer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(DCEFitViewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1772, 21))
        self.menubar.setObjectName("menubar")
        DCEFitViewer.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(DCEFitViewer)
        self.statusbar.setObjectName("statusbar")
        DCEFitViewer.setStatusBar(self.statusbar)

        self.retranslateUi(DCEFitViewer)
        QtCore.QMetaObject.connectSlotsByName(DCEFitViewer)

    def retranslateUi(self, DCEFitViewer):
        _translate = QtCore.QCoreApplication.translate
        DCEFitViewer.setWindowTitle(_translate("DCEFitViewer", "DCE fit viewer"))
        self.dceDir.setText(_translate("DCEFitViewer", "DCE dir"))
        self.label.setText(_translate("DCEFitViewer", "DCE dir"))
        self.dceDirSelectButton.setText(_translate("DCEFitViewer", "Select"))
        self.voxelsLabel.setText(_translate("DCEFitViewer", "Showing..."))
        self.previousButton.setText(_translate("DCEFitViewer", "Previous"))
        self.nextButton.setText(_translate("DCEFitViewer", "Next"))
        self.label_3.setText(_translate("DCEFitViewer", "Min C(t)"))
        self.label_2.setText(_translate("DCEFitViewer", "Max C(t)"))
from QbiPy.tools.mplwidget import MplWidget
