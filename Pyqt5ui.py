import sys
import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
import numpy as np
from scipy import ndimage


class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('test.ui', self)
        self.image = None
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave.triggered.connect(self.save_img)
        self.actionPrint.triggered.connect(self.createPrintDialog)
        self.actionQuit.triggered.connect(self.close)
        self.actionBig.triggered.connect(self.big_Img)
        self.actionSmall.triggered.connect(self.small_Img)

        #Chương 2
        self.actionRotation.triggered.connect(self.rotation)
        self.actionAffine.triggered.connect(self.affine)

        #Chương 3
        self.actioAnhXam.triggered.connect(self.anh_Xam)
        self.actionNegative.triggered.connect(self.anh_Negative)
        self.actionHistogram.triggered.connect(self.histogram_Equalization)
        self.actionLog.triggered.connect(self.Log) #Lỗi
        self.action2.triggered.connect(self.Gamma2)

        #Chương 4
        self.actionGaussan.triggered.connect(self.Gaussian)
        self.actionHigh_Boost.triggered.connect(self.High_Boost)
        self.actionLaplacian.triggered.connect(self.Laplacian)
        self.actionFilter_Average.triggered.connect(self.filter_Average)
        self.actionUnsharp.triggered.connect(self.Unsharp)

        #Chương 5
        self.actionTanSo.triggered.connect(self.Tan_so)
        self.actionIdeal_LPF.triggered.connect(self.imidlp)
        self.actionGaussian_HPF.triggered.connect(self.Gaussian_HighPass)

        #Chương 7
        self.actionDilate.triggered.connect(self.dilate)
        self.actionErode.triggered.connect(self.erode)
        self.actionOpen_2.triggered.connect(self.open)
        self.actionClose.triggered.connect(self.close_)
        self.actionHit_miss.triggered.connect(self.hitmis)#Lỗi
        self.actionGradient.triggered.connect(self.gradient)
        self.actionMorboundary.triggered.connect(self.morboundary)
        self.actionConvex.triggered.connect(self.convex)

        #Chương 8
        self.actionx_direcction_Sobel.triggered.connect(self.Sobel)
        self.actiony_direction_Sobel.triggered.connect(self.y_Sobel)
        self.actionLaplacian_2.triggered.connect(self.sobel_Laplacian)
        self.actionLaplacian_of_Gaussian.triggered.connect(self.lap_of_Gaussian)
        self.actionCanny.triggered.connect(self.Canny)

        #vùng set input
        self.dial.valueChanged.connect(self.rotation)
        self.horizontalSlider.valueChanged.connect(self.Gamma2)
        self.size_Img.valueChanged.connect(self.big_Img)
        self.gaussian_QSlider.valueChanged.connect(self.Gaussian)
        self.erosion.valueChanged.connect(self.erode)

        self.gray.stateChanged.connect(self.anh_Xam)

        self.hist.stateChanged.connect(self.histogram_Equalization)

        self.Qlog.valueChanged.connect(self.Log)

        self.sobel.stateChanged.connect(self.Sobel)
        self.sobel_x.stateChanged.connect(self.Sobel)
        self.sobel_y.stateChanged.connect(self.Sobel)

        self.cbLog.stateChanged.connect(self.Log)

        self.cbLap.stateChanged.connect(self.Laplacian)

        self.canny.stateChanged.connect(self.Canny)
        self.canny_min.valueChanged.connect(self.Canny)
        self.canny_max.valueChanged.connect(self.Canny)




    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        #BGR > RGB
        img = img.rgbSwapped()
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\DELL\PycharmProjects\DemoPro', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def save_img(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image)
        else:
            print("Error")

    def createPrintDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.imgLabel2.print_(printer)

    def big_Img(self , c):
        self.image = self.tmp
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.image = cv2.resize(self.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    ################# Chương 2 ##############################################################################
    def rotation(self, angle):
        self.image = self.tmp
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def affine(self):
        self.image = self.tmp
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))

        self.displayImage(2)

    ################# Chương 3 ##############################################################################
    def anh_Xam(self):
        self.image = self.tmp
        if self.gray.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.displayImage(2)

    def anh_Negative(self):
        self.image = self.tmp
        self.image = ~self.image
        self.displayImage(2)

    def histogram_Equalization(self):
        self.image = self.tmp
        if self.hist.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.equalizeHist(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        self.displayImage(2)

    def scaleImage(self, factor):
        self.scaleImage *= factor
        self.imgLabel.resize(self.scaleImage * self.imgLabel.pixmap() .size())

    def Log(self, c):
        self.image = self.tmp
        if self.cbLog.isChecked():
            img_2 = np.uint8(np.log(self.image))
            self.image = cv2.threshold(img_2, c, 225, cv2.THRESH_BINARY)[1]
        self.displayImage(2)

    def Gamma2(self , g):
        self.image = self.tmp
        c = 1
        gamma = [g]
        for i in range(len(gamma)):
            self.image = c * (self.image + 1) ** gamma[i]
        self.displayImage(2)

    ################# Chương 4 ##############################################################################
    def Gaussian(self , c):
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image, (5, 5), c)
        self.displayImage(2)

    def High_Boost(self):
        self.image = self.tmp
        if self.laplacian.ischecked():
            x = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            self.image = cv2.filter2D(np.array(self.image), -1, x)
        self.displayImage(2)

    def Laplacian(self):
        self.image = self.tmp
        h = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        if self.cbLap.isChecked():
            self.image = cv2.filter2D(np.array(self.image), -1, h)
        self.displayImage(2)

    def filter_Average(self):
        self.image = self.tmp

        self.image = cv2.medianBlur(self.image, 5)
        self.displayImage(2)

    def Unsharp(self):
        self.image = self.tmp
        gau = cv2.GaussianBlur(self.image, (9, 9), 10.0)
        self.image = cv2.addWeighted(self.image, 1.5, gau, -0.5, 0, self.image)
        self.displayImage(2)

    ################# Chương 5 ##############################################################################
    def Tan_so(self):
        self.image = self.tmp
        img_float32 = np.float32(self.image)
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        self.image = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        self.displayImage(2)

    def Ideal_LPF(self,sx, sy, d0):
        hr =(sx) / 2
        hc =(sy) / 2

        x = np.arange(-hc, hc)
        y = np.arange(-hr, hr)

        [x, y] = np.meshgrid(x, y)

        mg = np.sqrt(x ** 2 + y ** 2)
        return np.double(mg <= d0)

    def imidlp(self):
        self.image = self.tmp
        print(1)
        height, width, channels = self.image
        H = self.Ideal_LPF(height, width, d0=30)
        print(3)
        G = np.fft.fftshift(np.fft.fft2(self.image))
        print(4)
        Ip = G
        print(5)
        if len(G.shape) == 3:
            print(6)
            Ip[:, :, 0] = G[:, :, 0] = H
            Ip[:, :, 1] = G[:, :, 1] = H
            Ip[:, :, 2] = G[:, :, 2] = H
        else:
            print(7)
            Ip = G * H
            print(8)
        self.image = np.abs(np.fft.ifft2(np.fft.fftshift(Ip)), np.uint8)
        print(9)
        self.displayImage(2)
        print(10)

    def Gaussian_HighPass(self):
        self.image = self.tmp
        data = np.array(self.image, dtype=float)
        lowpass = ndimage.gaussian_filter(data, 3)
        self.image = data - lowpass
        self.displayImage(2)

    ################# Chương 7 ##############################################################################
    def dilate(self):
        self.image = self.tmp
        kernel = np.ones((2, 6), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        self.displayImage(2)

    def erode(self , iter):
        self.image = self.tmp
        if iter > 0 :
            kernel = np.ones((4, 7), np.uint8)
            self.image = cv2.erode(self.tmp, kernel, iterations=iter)
        else :
            kernel = np.ones((2, 6), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=iter*-1)
        self.displayImage(2)

    def open(self):
        self.image = self.tmp
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        self.displayImage(2)

    def close_(self):
        self.image = self.tmp
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        self.displayImage(2)

    #Lỗi
    def hitmis(self):
        self.image = self.tmp
        kernel = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_HITMISS, kernel)
        print(3)
        self.displayImage(2)
        print(4)

    def gradient(self):
        self.image = self.tmp
        kernel = np.ones((5, 5), np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)
        self.displayImage(2)

    def morboundary(self):
        self.image = self.tmp
        se = np.ones((3, 3), np.uint8)
        e1 = self.image - cv2.erode(self.image, se, iterations=1)
        self.image = e1
        self.displayImage(2)

    def convex(self):
        self.image = self.tmp
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(self.image, (3, 3))
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        # create an empty black image
        self.image = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0)  # green - color for contours
            color = (255, 0, 0)  # blue - color for convex hull
            # draw ith contour
            #cv2.drawContours( self.image, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours( self.image, hull, i, color, 1, 8)
        self.displayImage(2)

    ################# Chương 8 ##############################################################################
    def Sobel(self):
        if self.sobel.isChecked():
            self.image = self.tmp
            if self.sobel_x.isChecked():
                self.image = cv2.Sobel(self.image, cv2.CV_8U, 1, 0, ksize=5)
            if self.sobel_y.isChecked():
                self.image = cv2.Sobel(self.image, cv2.CV_8U, 0, 1, ksize=5)
        self.displayImage(2)
    def y_Sobel(self):
        self.image = self.tmp
        im = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.Sobel(im, cv2.CV_8U, 0, 1, ksize=5)

        self.displayImage(2)

    def sobel_Laplacian(self):
        self.image = self.tmp
        im = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.Laplacian(im, cv2.CV_8U)

        self.displayImage(2)

    def lap_of_Gaussian(self):
        self.image = self.tmp
        h = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        lap = cv2.filter2D(np.array(self.image), -1, h)
        kernal = np.array([[1, -2, 1, ], [-2, 4, -2], [1, -2, 1]])


        self.displayImage(2)
    def Canny(self ):
        self.image = self.tmp
        if self.canny.isChecked():
            can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(can, self.canny_min.value(), self.canny_max.value())
        self.displayImage(2)


app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())

