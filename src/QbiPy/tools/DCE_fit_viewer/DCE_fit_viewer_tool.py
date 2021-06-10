'''
GUI tool for inspecting tracer-kinetic model fits of individual DCE-MRI voxel time-series
'''

import sys
import os
import glob
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QGraphicsScene, QFileDialog
from PyQt5.QtCore import pyqtSlot 
from PyQt5.QtGui import QFont
import numpy as np

from QbiPy.image_io.analyze_format import read_analyze_img
from QbiPy.dce_models.data_io import get_dyn_vals

from QbiPy.tools.DCE_fit_viewer.DCE_fit_viewer import Ui_DCEFitViewer  

image_format = "*.hdr"

class DCEFitViewerTool(QMainWindow):

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    def __init__(self, DCE_dir=None, parent=None):

        #Create the UI
        QWidget.__init__(self, parent)
        self.ui = Ui_DCEFitViewer()
        self.ui.setupUi(self)
        self.setup_plot_axes()

        #Initialize instance variables
        self.voxel_offset = 0
        self.num_times = 0
        self.num_voxels = 0

        if DCE_dir == None:
            self.DCE_dir = ''
            self.select_DCE_dir()
        else:
            self.DCE_dir = DCE_dir
            self.ui.dceDirLineEdit.setText(DCE_dir)

        

    # Connect any signals that aren't auto name matched
    def connect_signals_to_slots(self):
        pass
        #QtCore.QObject.connect(self.ui.button_open,QtCore.SIGNAL("clicked()"), self.file_dialog)
    # --------------------------------------------------------------------
    #--------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    def setup_plot_axes(self):
        '''Add the named plot widgets from the UI into a list for convenience'''
        self.axes = []
        self.axes.append(self.ui.dcePlotWidget_1.canvas)
        self.axes.append(self.ui.dcePlotWidget_2.canvas)
        self.axes.append(self.ui.dcePlotWidget_3.canvas)
        self.axes.append(self.ui.dcePlotWidget_4.canvas)
        self.axes.append(self.ui.dcePlotWidget_5.canvas)
        self.axes.append(self.ui.dcePlotWidget_6.canvas)
        self.axes.append(self.ui.dcePlotWidget_7.canvas)
        self.axes.append(self.ui.dcePlotWidget_8.canvas)
        self.axes.append(self.ui.dcePlotWidget_9.canvas)
        self.axes.append(self.ui.dcePlotWidget_10.canvas)
        self.axes.append(self.ui.dcePlotWidget_11.canvas)
        self.axes.append(self.ui.dcePlotWidget_12.canvas)
        self.axes.append(self.ui.dcePlotWidget_13.canvas)
        self.axes.append(self.ui.dcePlotWidget_14.canvas)
        self.axes.append(self.ui.dcePlotWidget_15.canvas)


    #--------------------------------------------------------------------------
    # Auxilliary functions that control data
    #
    #--------------------------------------------------------------------------
    def load_data(self):
        #Get list of image names and update the relevant controls
        #Load in the initial subject
        if not os.path.isdir(self.DCE_dir):
            QMessageBox.warning(self, 'AIF directory not found!', 
                self.DCE_dir + ' is not a directory, check disk is connected')
            return

        self.num_times =len(glob.glob(
            os.path.join(self.DCE_dir, "Ct_sig", "Ct_sig" + image_format)))     
               
        if self.num_times:
            roi = read_analyze_img(
                os.path.join(self.DCE_dir, "ROI.hdr")) > 0
            self.Ct_s = get_dyn_vals(
                os.path.join(self.DCE_dir, "Ct_sig", "Ct_sig"), self.num_times, roi)
            self.Ct_m = get_dyn_vals(
                os.path.join(self.DCE_dir, "Ct_mod", "Ct_mod"), self.num_times, roi)
            self.sse = np.sqrt(np.sum((self.Ct_s - self.Ct_m)**2,1))
        else:
            QMessageBox.warning(self, 'No C(t) volumes found!', 
            'No contrast signal data found in ' + self.DCE_dir)

        self.num_voxels = self.Ct_s.shape[0]
        for offset in range(0, self.num_voxels, 15):
            v1 = offset+1
            v2 = min(offset+15, self.num_voxels)
            vox_str = f"Voxels {v1} to {v2}"
            self.ui.voxelComboBox.addItem(vox_str)

        dce_dir = os.path.split(self.DCE_dir)[1]
        self.ui.dceDir.setText(dce_dir)
        self.ui.dceDir.setFont(QFont("Arial", 12, QFont.Bold))

        self.update_voxel_display()
    

    #--------------------------------------------------------------------------
    def update_voxel_display(self):
        
        min_Ct = self.ui.minCtSpinBox.value()
        max_Ct = self.ui.maxCtSpinBox.value()

        for i_vox in range(15):
            voxel = i_vox + self.voxel_offset
            self.axes[i_vox].ax.clear()

            if 0 <= voxel < self.num_voxels:
                self.axes[i_vox].ax.plot(self.Ct_s[voxel,:])
                self.axes[i_vox].ax.plot(self.Ct_m[voxel,:], 'r')

                self.axes[i_vox].ax.set_xlabel('# timepoint')
                self.axes[i_vox].ax.set_ylabel('C(t)')

                sse_str = f'Voxel {voxel+1}: SSE = {self.sse[voxel]:4.3f}'
                self.axes[i_vox].ax.set_title(sse_str)

                ymin, ymax = self.axes[i_vox].ax.get_ylim()
                if min_Ct:
                    ymin = min_Ct
                if max_Ct:
                    ymax = max_Ct

                self.axes[i_vox].ax.set_ylim(ymin, ymax)

            self.axes[i_vox].draw()

        v1 = self.voxel_offset + 1
        v2 = min(self.voxel_offset + 15, self.num_voxels)
        v_str = f'Showing voxels {v1} to {v2} of {self.num_voxels}'
        self.ui.voxelsLabel.setText(v_str)

        self.ui.previousButton.setEnabled(self.voxel_offset >= 15)
        self.ui.nextButton.setEnabled(self.voxel_offset < (self.num_voxels-15))

    #--------------------------------------------------------------------------
    def select_DCE_dir(self):
        self.ui.aifDirLineEdit.setEnabled(False)
        temp_dir = QFileDialog.getExistingDirectory(self, 'Select the AIF directory',
            self.DCE_dir,
            QFileDialog.ShowDirsOnly)
        
        if temp_dir:
            self.DCE_dir = temp_dir
            self.ui.dceDirLineEdit.setText(temp_dir)
            if not self.dynamic_dir:
                self.dynamic_dir = self.DCE_dir
        
        self.ui.dceDirLineEdit.setEnabled(True)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # UI Callbacks
    # We make use of QT's autoconnect naming feature here so we don't need to
    #   explicitly connect the various widgets with their callbacks
    #--------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    @pyqtSlot()
    def on_dceDirSelectButton_clicked(self):
        self.select_DCE_dir()
        self.load_data(0)
        
    # --------------------------------------------------------------------
    @pyqtSlot()
    def on_nextButton_clicked(self):
        next_vox = self.voxel_offset + 15
        if 0 <= next_vox < self.num_voxels:
            self.voxel_offset = next_vox
            self.ui.voxelComboBox.setCurrentIndex(self.voxel_offset/15)
            self.update_voxel_display()

    # --------------------------------------------------------------------
    @pyqtSlot()
    def on_previousButton_clicked(self):
        next_vox = self.voxel_offset - 15
        if 0 <= next_vox < self.num_voxels:
            self.voxel_offset = next_vox
            self.ui.voxelComboBox.setCurrentIndex(self.voxel_offset/15)
            self.update_voxel_display(self.voxel_offset/15)

    # --------------------------------------------------------------------
    def on_voxelComboBox_activated(self):
        self.voxel_offset = self.ui.voxelComboBox.currentIndex()*15
        self.update_voxel_display()

    # --------------------------------------------------------------------
    def on_minCtSpinBox_valueChanged(self, value:float):
        self.update_voxel_display()

    # --------------------------------------------------------------------
    def on_maxCtSpinBox_valueChanged(self, value:float):
        self.update_voxel_display()

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #---------------------- END OF CLASS -----------------------------------
    #--------------------------------------------------------------------------

#--------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)

    DCE_dir = None
    
    if len(sys.argv) > 1:
        DCE_dir = sys.argv[1]

    myapp = DCEFitViewerTool(DCE_dir)
    myapp.show()
    myapp.load_data()

    sys.exit(app.exec_())