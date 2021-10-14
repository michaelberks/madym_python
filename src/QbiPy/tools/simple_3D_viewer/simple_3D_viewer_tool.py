'''
GUI tool for inspecting 3D image volumes stored in Analyze 7.5 format
'''

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

from QbiPy.tools.simple_3D_viewer.simple_3D_viewer import Ui_Simple3DViewer      

image_format = "*.hdr"

class Simple3DViewerTool(QMainWindow):

        #Set constants/variables that persist for all user sessions:
        #color1 = [1 1 1]
        #color2 = [212 208 200]/255
        #color3 = [0 0 0]

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    def __init__(self, image_dir=None, parent=None):

        #Create the UI
        QWidget.__init__(self, parent)
        self.ui = Ui_Simple3DViewer()
        self.ui.setupUi(self)
        self.ui.scene1 = qs.QbiQscene()
        self.ui.leftGraphicsView.setScene(self.ui.scene1)
        self.ui.colorbar = qs.QbiQscene()
        self.ui.colorbarGraphicsView.setScene(self.ui.colorbar)

        #Initialize instance variables
        self.image_names = []
        self.num_images = 0
        self.curr_image = 0
        self.images = []

        self.num_slices = 0
        self.curr_slice = 0

        if image_dir == None:
            self.image_dir = ''
            self.select_image_dir()
        else:
            self.image_dir = image_dir
            self.ui.imageDirLineEdit.setText(image_dir)

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
    def get_image_list(self, init_image=0):
        #Get list of image names and update the relevant controls
        #Load in the initial subject
        if not os.path.isdir(self.image_dir):
            QMessageBox.warning(self, 'Image directory not found!', 
                self.image_dir + ' is not a directory, check disk is connected')
            return

        self.image_names = [os.path.basename(f) for f in glob.glob(
            os.path.join(self.image_dir, image_format))]
        self.num_images = len(self.image_names)       
               
        if self.num_images:
            if init_image and init_image <= self.num_images:
                self.curr_image = init_image
            else:
                self.curr_image = 0
            
            
            #Update uicontrols now we have data
            self.ui.imageComboBox.setEnabled(True)
            self.ui.imageComboBox.clear()
            image_text = 'Images in folder:'
            for image in self.image_names:
                self.ui.imageComboBox.addItem(image)
                image_text += '\n'
                image_text += image
                
            self.ui.imageListTextEdit.setText(image_text)

            #Load in the images
            self.load_images()

            #Load in the images for the first pair and update the pair
            #selecter
            self.update_curr_image()
            
        else:
            QMessageBox.warning(self, 'No subjects found!', 'No subjects found in ' + self.image_dir)
                
    
    #--------------------------------------------------------------------------       
    def load_images(self):
        #h = waitbar(0,'Loading MR volumes. Please wait...')
        self.images = []
        for img_name in  self.image_names:
            img_path = os.path.join(self.image_dir, img_name)

            img = read_analyze_img(img_path)
            self.images.append(img)
        
        if self.curr_image >= self.num_images:
            self.curr_image = self.num_images-1

    #----------------------------------------------------------------------
    def update_curr_image(self):
        image_text = 'Select image: ' + str(self.curr_image+1) + ' of ' + str(self.num_images)
        self.ui.selectImageLabel.setText(image_text)
        
        self.ui.imageComboBox.setCurrentIndex(self.curr_image)
        self.ui.nextImageButton.setEnabled(self.curr_image < self.num_images-1)
        self.ui.previousImageButton.setEnabled(self.curr_image)

        #Get size of this image
        self.num_slices = self.images[self.curr_image].shape[2]

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
    def update_volume_display(self):
           
        self.ui.scene1.reset()
        image = self.images[self.curr_image]

        #Get current slice of each volume
        slice = image[:,:,self.curr_slice]
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
        
        self.ui.volumeLabel.setText('%s: slice %d' 
            %(self.image_names[self.curr_image], self.curr_slice+1))
        self.ui.selectSliceLabel.setText('Select slice: %d of %d' 
            %(self.curr_slice+1, self.num_slices))

    #--------------------------------------------------------------------------
    def select_image_dir(self):
        self.ui.imageDirLineEdit.setEnabled(False)
        temp_dir = QFileDialog.getExistingDirectory(self, 'Select the image directory',
            self.image_dir,
            QFileDialog.ShowDirsOnly)
        
        if temp_dir:
            self.image_dir = temp_dir
            self.ui.imageDirLineEdit.setText(temp_dir)
        
        self.ui.imageDirLineEdit.setEnabled(True)

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
    def on_imageDirSelectButton_clicked(self):
        self.select_image_dir()
        self.get_image_list(0)

        
    # --------------------------------------------------------------------
    @pyqtSlot()
    def on_nextImageButton_clicked(self):
        next_image = self.curr_image + 1
        if 0 <= next_image < self.num_images:
            self.curr_image = next_image
            self.update_curr_image()

    # --------------------------------------------------------------------
    @pyqtSlot()
    def on_previousImageButton_clicked(self):
        next_image = self.curr_image - 1
        if 0 <= next_image < self.num_images:
            self.curr_image = next_image
            self.update_curr_image()

    # --------------------------------------------------------------------
    def on_imageComboBox_activated(self):
        self.curr_image = self.ui.imageComboBox.currentIndex()
        self.update_curr_image()

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
    #             if self.curr_image < self.num_images
    #                 self.curr_image = self.curr_image + 1
    #                 update_dynamic = true
                

    #         case 'leftarrow'
    #             if self.curr_image > 1
    #                 self.curr_image = self.curr_image - 1
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
    #         set(ui.dynamic_slider, 'value', self.curr_image)
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

    study_dir = None
    init_image = 0
    if len(sys.argv) > 1:
        study_dir = sys.argv[1]
    if len(sys.argv) > 2:
        init_image = int(sys.argv[2])

    myapp = Simple3DViewerTool(study_dir)
    myapp.show()
    myapp.get_image_list(init_image)
    sys.exit(app.exec_())