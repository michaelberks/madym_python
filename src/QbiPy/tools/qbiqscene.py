'''
Subclass of Qt's QGraphicsScene to enable custom annotations eg tumour borders
'''

from PyQt5.QtGui import QPixmap, QImage, qRgb, QPainterPath, QPen, QBrush, QPainter
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPathItem
from PyQt5.QtCore import QRectF, Qt, QPointF

class QbiQscene(QGraphicsScene):
    def __init__(self):
        #Inherit from QGraphicsScen
        QGraphicsScene.__init__(self)
        
        #QImage*
        self.image = QImage()

        # QGraphicsPixmapItem* Pixmap item for raw image in scene
        self.raw_pixmap_item = self.addPixmap( QPixmap.fromImage(self.image) )

        #Initialize colour table
        self.update_raw_color_table(0, 255)

        #Create empty container for annotations
        self.annotations = []
        self.annotation_points = []

    def reset(self):
        for an in self.annotations:
            self.removeItem(an)
        self.annotations = []
        self.image = QImage()
        self.update_raw_color_table(0, 255)

    def update_raw_color_table(self, contrast_min=0, contrast_max=255):
        self.raw_colour_table = []

        #fill everything from first to constrast_min_ with 0 (black)
        self.raw_colour_table = [qRgb(0,0,0) for i in range(contrast_min)]

        #fill values in between with a linear gradient
        denom = contrast_max - contrast_min
        for i in range(contrast_min,contrast_max):
            grey =  255*(i - contrast_min) / denom
            self.raw_colour_table += [qRgb(grey, grey, grey)]

        #fill everything from constrast_max_ to end with 255 (white)
        self.raw_colour_table += [qRgb(255,255,255) for i in range(contrast_max, 256)]
        self.update_pixmap()

    def update_pixmap(self, win_width = 0, win_height = 0):
        self.image.setColorTable(self.raw_colour_table)
        
        if win_width:
            image = QPixmap.fromImage(self.image)
            scaled_img = image.scaled(win_width, win_height, Qt.IgnoreAspectRatio)
            self.raw_pixmap_item.setPixmap( scaled_img )
        else:
            self.raw_pixmap_item.setPixmap( QPixmap.fromImage(self.image) )
            self.raw_pixmap_item.setPos(-self.image.width()/2, -self.image.height()/2)
            self.setSceneRect(QRectF(-self.image.width()/2, -self.image.height()/2,
                                self.image.width(), self.image.height() ) )
        
            

    def set_image(self, img, win_width = 0, win_height = 0):
        self.image = img
        self.update_pixmap(win_width, win_height)

    def add_annotation(self, x_pts, y_pts, pen = QPen(Qt.blue, 2.0), brush = QBrush(Qt.SolidPattern)):
        path = QPainterPath()
        n_points = len(x_pts)
        if not n_points or n_points != len(y_pts):
            return

        path.moveTo(QPointF(x_pts[0],y_pts[0]))
        for i in range(1,n_points):
            path.lineTo(QPointF(x_pts[i],y_pts[i]))

        pathItem = self.addPath(path, pen , brush)
        self.annotations.append(pathItem)

    def clear_annotation_points(self):
        for item in self.annotation_points:
            self.removeItem(item)

        self.annotation_points.clear()

    def add_annotation_points(self, x_pts, y_pts, pen = QPen(Qt.red, 0.5), brush = QBrush(Qt.SolidPattern)):
        self.clear_annotation_points()
        for x,y in zip(x_pts,y_pts):
            point = self.addEllipse(x, y, 0.5, 0.5, pen, brush)
            self.annotation_points.append(point)

    def update_annotation(self, anno_idx, x_pts, y_pts):
        if anno_idx >= len(self.annotations):
            return
        path = QPainterPath()
        
        n_points = len(x_pts)
        if not n_points or n_points != len(y_pts):
            return

        path.moveTo(QPointF(x_pts[0],y_pts[0]))
        for i in range(1,n_points):
            path.lineTo(QPointF(x_pts[i],y_pts[i]))

        self.annotations[anno_idx].setPath(path)

    def show_annotation(self, anno_idx, is_visible):
        if anno_idx >= len(self.annotations):
            return
        self.annotations[anno_idx].setVisible(is_visible)

