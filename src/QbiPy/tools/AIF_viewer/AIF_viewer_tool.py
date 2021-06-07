import sys
import os
import glob
from PyQt5.QtWidgets import QApplication, QPushButton, QAction, QMainWindow, QWidget, QMessageBox, QGraphicsScene, QFileDialog
from PyQt5.QtCore import QObject, Qt, pyqtSlot 
from PyQt5.QtGui import QImage, qRgb
import numpy as np
from scipy import ndimage

from QbiPy.image_io.analyze_format import read_analyze_img
from QbiPy.tools import qbiqscene as qs

from QbiPy.tools.AIF_viewer.AIF_viewer import Ui_AIFViewer    
from QbiPy.dce_models.dce_aif import Aif, AifType  

image_format = "*.hdr"
AIF_format = "*_AIF.txt"

class AIFViewerTool(QMainWindow):

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    def __init__(self, AIF_dir=None, dynamic_image=None, parent=None):

        #Create the UI
        QWidget.__init__(self, parent)
        self.ui = Ui_AIFViewer()
        self.ui.setupUi(self)
        self.ui.scene1 = qs.QbiQscene()
        self.ui.leftGraphicsView.setScene(self.ui.scene1)
        self.ui.colorbar = qs.QbiQscene()
        self.ui.colorbarGraphicsView.setScene(self.ui.colorbar)

        #Initialize instance variables
        self.AIF_names = []
        self.num_AIFs = 0
        self.curr_AIF = 0
        self.AIFs = []
        self.AIF_masks = []

        self.num_slices = 0
        self.curr_slice = 0

        if AIF_dir == None:
            self.AIF_dir = ''
            self.select_AIF_dir()
        else:
            self.AIF_dir = AIF_dir
            self.ui.aifDirLineEdit.setText(AIF_dir)

        if dynamic_image is None:
            self.dynamic_image_path = ''
            self.select_dynamic_image()
        else:
            self.dynamic_image_path = dynamic_image
            self.ui.dynVolLineEdit.setText(dynamic_image)

    # Connect any signals that aren't auto name matched
    def connect_signals_to_slots(self):
        pass
        #QtCore.QObject.connect(self.ui.button_open,QtCore.SIGNAL("clicked()"), self.file_dialog)
    # --------------------------------------------------------------------
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Auxilliary functions that control data
    #
    #--------------------------------------------------------------------------
    def get_AIF_list(self, init_AIF=0):
        #Get list of image names and update the relevant controls
        #Load in the initial subject
        if not os.path.isdir(self.AIF_dir):
            QMessageBox.warning(self, 'AIF directory not found!', 
                self.AIF_dir + ' is not a directory, check disk is connected')
            return

        self.AIF_names = [os.path.basename(f) for f in glob.glob(
            os.path.join(self.AIF_dir, AIF_format))]
        self.num_AIFs = len(self.AIF_names)       
               
        if self.num_AIFs:
            if init_AIF and init_AIF <= self.num_AIFs:
                self.curr_AIF = init_AIF
            else:
                self.curr_AIF = 0
            
            
            #Update uicontrols now we have data
            self.ui.aifComboBox.setEnabled(True)
            self.ui.aifComboBox.clear()
            AIF_text = 'AIFs in folder:'
            for aif in self.AIF_names:
                self.ui.aifComboBox.addItem(aif)
                AIF_text += '\n'
                AIF_text += aif
                
            self.ui.aifInfoTextEdit.setText(AIF_text)

            #Load in the images
            self.load_AIFs()

            #Load in the images for the first pair and update the pair
            #selecter
            self.update_curr_AIF()
            
        else:
            QMessageBox.warning(self, 'No subjects found!', 'No subjects found in ' + self.AIF_dir)
                
    #--------------------------------------------------------------------------
    def load_dynamic_image(self):
        if not os.path.isfile(self.dynamic_image_path):
            QMessageBox.warning(self, 'Dynamic image not found!', 
                self.dynamic_image_path + ' does not exist, check disk is connected')
            return

        self.dynamic_image = read_analyze_img(self.dynamic_image_path)

        #Get size of this image
        self.num_slices = self.dynamic_image.shape[2]

        #By default stay on the same previous slice as before
        if self.curr_slice >= self.num_slices:
            self.curr_slice = self.num_slices-1

        #Set slices slider
        if self.num_slices > 1:
            self.ui.sliceSlider.setRange(1, self.num_slices)
            self.ui.sliceSlider.setSingleStep(1)
            self.ui.sliceSlider.setValue(self.curr_slice+1)
            self.ui.sliceSlider.setEnabled(True)        

        #Display image
        self.update_volume_display()

        #Make colorbar
        self.make_colorbar()
        
    #--------------------------------------------------------------------------       
    def load_AIFs(self):
        #h = waitbar(0,'Loading MR volumes. Please wait...')
        self.AIFs = []
        self.AIF_masks = []
        for AIF_name in  self.AIF_names:
            aif_path = os.path.join(self.AIF_dir, AIF_name)
            aif = Aif(aif_type=AifType.FILE, filename=aif_path)
            self.AIFs.append(aif)

            aif_mask_name = os.path.splitext(aif_path)[0] + ".hdr"
            if os.path.isfile(aif_mask_name):
                aif_mask = read_analyze_img(aif_mask_name)
                self.AIF_masks.append(aif_mask==1)
            else:
                self.AIF_masks.append(None)
                print('Missing mask ', aif_mask_name)
        
        if self.curr_AIF >= self.num_AIFs:
            self.curr_AIF = self.num_AIFs-1

    #----------------------------------------------------------------------
    def update_curr_AIF(self):
        AIF_text = 'Select AIF: ' + str(self.curr_AIF+1) + ' of ' + str(self.num_AIFs)
        self.ui.selectAifLabel.setText(AIF_text)
        
        self.ui.aifComboBox.setCurrentIndex(self.curr_AIF)
        self.ui.nextAifButton.setEnabled(self.curr_AIF < self.num_AIFs-1)
        self.ui.previousAifButton.setEnabled(self.curr_AIF)
        
        #Display image
        self.update_AIF_display()

    #--------------------------------------------------------------------------
    def update_AIF_display(self):
        aif = self.AIFs[self.curr_AIF]
        
        self.ui.aifPlotWidget.canvas.ax.clear()
        self.ui.aifPlotWidget.canvas.ax.plot(
                aif.times_, aif.base_aif_)
        self.ui.aifPlotWidget.canvas.ax.set_xlabel(
            'Time (mins)')
        self.ui.aifPlotWidget.canvas.ax.set_ylabel(
            'C(t)')
        self.ui.aifPlotWidget.canvas.ax.set_title(
            self.AIF_names[self.curr_AIF])
        self.ui.aifPlotWidget.canvas.draw()

    #--------------------------------------------------------------------------
    def update_volume_display(self):
           
        self.ui.scene1.reset()

        #Get current slice of each volume
        slice = self.dynamic_image[:,:,self.curr_slice]
        self.slice_min = np.min(slice[np.isfinite(slice)])
        self.slice_max = np.max(slice[np.isfinite(slice)])
        
        if self.slice_min == self.slice_max:
            self.slice_range = 1
        else:
            self.slice_range = self.slice_max - self.slice_min

        scaled_slice = (255*(slice-self.slice_min) / self.slice_range).astype(np.uint8)

        #Compute the apsect ratios for these images (they may vary from
        #pair to pair)
        height,width = slice.shape

        #Compute map limits for color scaling
        min_contrast = 0
        max_contrast = 255

        self.ui.minContrastSlider.setRange(min_contrast, max_contrast-1)
        self.ui.minContrastSlider.setSingleStep(1)
        self.ui.minContrastSlider.setValue(min_contrast)
        self.ui.minContrastSlider.setEnabled(True)

        self.ui.maxContrastSlider.setRange(min_contrast+1, max_contrast)
        self.ui.maxContrastSlider.setSingleStep(1)
        self.ui.maxContrastSlider.setValue(max_contrast)
        self.ui.maxContrastSlider.setEnabled(True)

        self.set_contrast_label(min_contrast, max_contrast)

        #Make the maps visible
        self.ui.scene1.update_raw_color_table(min_contrast, max_contrast)
        q_img1 = QImage(scaled_slice.data, width, height, QImage.Format_Indexed8)
        self.ui.scene1.set_image(q_img1)
        self.ui.leftGraphicsView.fitInView(self.ui.scene1.itemsBoundingRect(), 
                            Qt.KeepAspectRatio)
        self.ui.scene1.update() 

        #Add annotation if any
        aif_mask = self.AIF_masks[self.curr_AIF]
        if not aif_mask is None:
            aif_mask_slice = aif_mask[:,:,self.curr_slice]
            aif_xyz = np.nonzero(aif_mask_slice)
            if len(aif_xyz[0]):
                self.ui.scene1.add_annotation_points(aif_xyz[1] - width/2, aif_xyz[0] - height/2)
            else:
                self.ui.scene1.clear_annotation_points()
        
        self.ui.dynVolLabel.setText('%s: slice %d' 
            %(
                os.path.basename(self.dynamic_image_path),
                self.curr_slice+1))
        self.ui.selectSliceLabel.setText('Select slice: %d of %d' 
            %(self.curr_slice+1, self.num_slices))

    #--------------------------------------------------------------------------
    def select_dynamic_image(self):
        self.ui.dynVolLineEdit.setEnabled(False)
        temp_path = QFileDialog.getOpenFileName(self, 'Open file', 
            self.dynamic_dir, "Image files (*.hdr)")
        
        if temp_path:
            self.dynamic_dir = os.path.dirname(temp_path)
            self.dynamic_image_path = temp_path
            self.ui.dynVolLineEdit.setText(temp_path)
        
        self.ui.dynVolLineEdit.setEnabled(True)

    #--------------------------------------------------------------------------
    def select_AIF_dir(self):
        self.ui.aifDirLineEdit.setEnabled(False)
        temp_dir = QFileDialog.getExistingDirectory(self, 'Select the AIF directory',
            self.AIF_dir,
            QFileDialog.ShowDirsOnly)
        
        if temp_dir:
            self.AIF_dir = temp_dir
            self.ui.aifDirLineEdit.setText(temp_dir)
            if not self.dynamic_dir:
                self.dynamic_dir = self.AIF_dir
        
        self.ui.aifDirLineEdit.setEnabled(True)

    #--------------------------------------------------------------------------
    def set_contrast_label(self, min_contrast, max_contrast):
        min_val = self.slice_range*min_contrast/255 + self.slice_min
        max_val = self.slice_range*max_contrast/255 + self.slice_min
        self.ui.minContrast.setText('%g' %(min_val))
        self.ui.maxContrast.setText('%g' %(max_val))
        self.ui.minContrast1.setText('%g' %(min_val))
        self.ui.maxContrast1.setText('%g' %(max_val))

    def make_colorbar(self):
        self.ui.colorbarGraphicsView.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.ui.colorbarGraphicsView.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)

        self.ui.colorbar.reset()
        self.ui.colorbar.update_raw_color_table(0, 255)
        colorbar = np.repeat(np.expand_dims(np.arange(255),0), 1, 0).astype(np.uint8)
        height,width = colorbar.shape
        q_img = QImage(colorbar.data, width, height, QImage.Format_Indexed8)
        
        self.ui.colorbar.set_image(q_img)
        self.ui.colorbarGraphicsView.fitInView(self.ui.colorbar.sceneRect(), 
                            Qt.IgnoreAspectRatio)
        self.ui.colorbar.update()
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # UI Callbacks
    # We make use of QT's autoconnect naming feature here so we don't need to
    #   explicitly connect the various widgets with their callbacks
    #--------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    @pyqtSlot()
    def on_aifDirSelectButton_clicked(self):
        self.select_AIF_dir()
        self.get_AIF_list(0)

    # -------------------------------------------------------------------------
    @pyqtSlot()
    def on_dynVolDirSelectButton_clicked(self):
        self.select_dynamic_image()
        self.load_dynamic_image()
        
    # --------------------------------------------------------------------
    @pyqtSlot()
    def on_nextAifButton_clicked(self):
        next_aif = self.curr_AIF + 1
        if 0 <= next_aif < self.num_AIFs:
            self.curr_AIF = next_aif
            self.update_curr_AIF()

    # --------------------------------------------------------------------
    @pyqtSlot()
    def on_previousAifButton_clicked(self):
        next_aif = self.curr_AIF - 1
        if 0 <= next_aif < self.num_AIFs:
            self.curr_AIF = next_aif
            self.update_curr_AIF()

    # --------------------------------------------------------------------
    def on_aifComboBox_activated(self):
        self.curr_AIF = self.ui.aifComboBox.currentIndex()
        self.update_curr_AIF()

    # --------------------------------------------------------------------
    def on_minContrastSlider_sliderMoved(self, value:int):
        min_slider = int(value)
        max_slider = max(min_slider + 1, self.ui.maxContrastSlider.value())
        
        self.ui.maxContrastSlider.setValue(max_slider)
        self.ui.scene1.update_raw_color_table(min_slider, max_slider)
        self.set_contrast_label(min_slider, max_slider)

    # --------------------------------------------------------------------
    def on_maxContrastSlider_sliderMoved(self, value:int):
        max_slider = int(value)
        min_slider = min(max_slider-1, self.ui.minContrastSlider.value())
        
        self.ui.minContrastSlider.setValue(min_slider)
        self.ui.scene1.update_raw_color_table(min_slider, max_slider)
        self.set_contrast_label(min_slider, max_slider)

    # --------------------------------------------------------------------
    def on_sliceSlider_sliderMoved(self, value:int):
        next_slice = int(value)-1
        if 0 <= next_slice < self.num_slices:
            self.curr_slice = next_slice
            self.update_volume_display()

    # --------------------------------------------------------------------
    @pyqtSlot()
    def wheelEvent(self,event):
        delta = event.angleDelta().y()
        step = (delta and delta // abs(delta))
        next_slice = self.curr_slice + step
        if 0 <= next_slice < self.num_slices:
            self.curr_slice = next_slice
            self.ui.sliceSlider.setValue(self.curr_slice+1)
            self.update_volume_display()
            

    #---------------------------------------------------------------------
    # def on_keypress_Callback(self, eventdata):
        
    #     update_dynamic = false
    #     update_slice = false
    #     switch eventdata.Key
    #         case 'rightarrow'
    #             if self.curr_AIF < self.num_AIFs
    #                 self.curr_AIF = self.curr_AIF + 1
    #                 update_dynamic = true
                

    #         case 'leftarrow'
    #             if self.curr_AIF > 1
    #                 self.curr_AIF = self.curr_AIF - 1
    #                 update_dynamic = true
                

    #         case 'uparrow'
    #             if self.curr_slice < self.num_slices
    #                 self.curr_slice = self.curr_slice + 1
    #                 update_slice = true
                

    #         case 'downarrow'
    #             if self.curr_slice > 1
    #                 self.curr_slice = self.curr_slice - 1
    #                 update_slice = true
           
    #     if update_slice
    #         set(ui.slice_slider, 'value', self.curr_slice)
    #         update_volume_display

    #     if update_dynamic
    #         set(ui.dynamic_slider, 'value', self.curr_AIF)
    #         update_dynamic_display

    # #---------------------------------------------------------------------
    # def on_scroll_Callback(hObject, eventdata) ##ok
    # # Callback...
    #     if self.num_slices
    #         self.curr_slice = min(max(self.curr_slice + eventdata.VerticalScrollCount,1),self.num_slices)
    #         set(ui.slice_slider, 'value', self.curr_slice)
    #         update_volume_display
        
    

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #---------------------- END OF CLASS -----------------------------------
    #--------------------------------------------------------------------------

#--------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)

    aif_dir = None
    init_image = 0
    dynamic_image = None
    if len(sys.argv) > 1:
        aif_dir = sys.argv[1]
    if len(sys.argv) > 2:
        dynamic_image = sys.argv[2]

    myapp = AIFViewerTool(aif_dir, dynamic_image)
    myapp.show()
    myapp.get_AIF_list(init_image)
    myapp.load_dynamic_image()

    sys.exit(app.exec_())