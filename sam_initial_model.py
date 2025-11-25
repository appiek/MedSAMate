# -*- coding: utf-8 -*-
# ---------------------------------------------------------
from PyQt5.QtCore import QPoint, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget, QMessageBox, QCheckBox, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# ---------------------------------------------------------
import glob
import os
import sys
from argparse import Namespace
import matplotlib.pyplot as plt
import pyautogui
import numpy as np
import nibabel as nib
from globalVariables import *
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from volume import volume, render_update
import vtk.util.numpy_support as numpy_support
from callback import CallBack
from interactor import *
# -------------------------------------------------------------
import cv2
import time
import medpy
import trimesh
import math
from random import shuffle
from medpy.io import load, save
from scipy.io import savemat, loadmat
import pydicom
from skimage.measure import marching_cubes
from scipy import ndimage
# --------------------------------------------------------------
import torch
import torch.nn as nn
# --------------------------------------------------------------
from sam_med2d_funcs import *
import resources_rc

error = vtk.vtkOutputWindow()
error.SetGlobalWarningDisplay(0)  # 关闭vtk报错信息


def BoundingBox_Coordinate(Seg, offset=20):
    H, W, C = Seg.shape
    Mask = np.sum(Seg > 0, 2)
    Mask_H = np.sum(Mask, 1) > 0
    indx_H = np.where(Mask_H)
    Mask_V = np.sum(Mask, 0) > 0
    indx_V = np.where(Mask_V)

    TL_x = max(np.min(indx_H) - offset, 0)
    TL_y = max(np.min(indx_V) - offset, 0)
    BR_x = min(np.max(indx_H) + offset, H)
    BR_y = min(np.max(indx_V) + offset, W)

    Mask = np.sum(np.sum(Seg > 0, 0), 0)
    mask_id_list = np.where(Mask > 0)
    mask_id_low = min(np.max(mask_id_list) + offset, C)
    mask_id_up = max(np.min(mask_id_list) - offset, 0)

    return TL_x, BR_x, TL_y, BR_y, mask_id_up, mask_id_low


def savestl(mask, spacing, subjname, savepath, smoothfactor=2.0):
    TL_x, BR_x, TL_y, BR_y, mask_id_up, mask_id_low = BoundingBox_Coordinate(mask)
    seg = np.float32(mask[TL_x:BR_x, TL_y: BR_y, mask_id_up:mask_id_low] > 0)
    seg = ndimage.filters.gaussian_filter(seg, sigma=smoothfactor, truncate=8.0)
    Seg = np.zeros(mask.shape)
    Seg[TL_x:BR_x, TL_y: BR_y, mask_id_up:mask_id_low] = seg

    verts, faces, normals, values = marching_cubes(Seg, level=0.5, spacing=spacing, step_size=1.0)
    # ================save the surface=====================
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export(savepath + subjname + '.stl')


def convertNsave(dicom_file_ori, file_dicom, file_dir, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    """
    dicom_file = pydicom.dcmread(file_dicom)
    arr = dicom_file_ori.pixel_array
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.SliceThickness = dicom_file_ori.SliceThickness
    dicom_file.PixelSpacing = dicom_file_ori.PixelSpacing
    dicom_file.PixelData = arr.tobytes()
    dicom_file.InstanceNumber = str(index + 1)
    dicom_file.save_as(os.path.join(file_dir, f'{"{:04d}".format(index + 1)}.dcm'))


def Load_IM0_BIM(inputfile):
    # ----------------------------------------------------------------------------------
    rootpath = '.'
    # ============IM0 data transform into matlab===========
    slicenumber = os.popen('get_slicenumber ' + inputfile).read()
    slicenumber = slicenumber.split(' ')[1]
    os.system('exportMath ' + inputfile + ' matlab ' + rootpath + '\\temp.mat' + ' 0 ' + slicenumber)
    input_mridata = loadmat(rootpath + '\\temp.mat')['scene']
    os.remove(rootpath + '\\temp.mat')
    return input_mridata


def Save_BIM(Img, output_file, input_file=None):
    # ----------------------------------------------------------------------------------
    rootpath = '.'
    img_shape = np.shape(Img)
    if input_file == None:
        savemat(rootpath + '\\temp.mat', {'scene': np.uint8(Img)})
        os.system(
            'importMath ' + rootpath + '/temp.mat ' + 'matlab ' + output_file + ' ' + str(img_shape[0]) + ' ' + str(
                img_shape[1]) + ' ' + str(img_shape[2]))
        os.remove(rootpath + '\\temp.mat')
    else:
        savemat(rootpath + '\\temp.mat', {'scene': np.uint8(Img)})
        os.system('importMath ' + rootpath + '\\temp.mat ' + 'matlab ' + rootpath + '\\temp.BIM ' + str(
            img_shape[0]) + ' ' + str(img_shape[1]) + ' ' + str(img_shape[2]))
        os.system('ndthreshold ' + rootpath + '\\temp.BIM ' + rootpath + '\\temp2.BIM 0 1 1')
        os.system('copy_pose ' + rootpath + '\\temp2.BIM ' + input_file + ' ' + output_file)
        os.remove(rootpath + '\\temp.BIM')
        os.remove(rootpath + '\\temp2.BIM')
        os.remove(rootpath + '\\temp.mat')


def LevelAndWidth(self):
    scalarRange = self.reader.GetOutput().GetScalarRange()
    print('scalarRange', scalarRange)
    if (scalarRange[0] < -1e10) | (scalarRange[0] > -1e10) | (scalarRange[1] < -1e10) | (scalarRange[1] > -1e10):
        scalarRange = (0, 4095)
    window = scalarRange[1] - scalarRange[0]
    level = (scalarRange[0] + scalarRange[1]) / 5.0
    return window, level


def polar360(x_input, y_input, x_ori=0, y_ori=0):
    x = x_input - x_ori
    y = y_input - y_ori
    rdius = math.hypot(y, x)
    theta = math.degrees(math.atan2(x, y)) + (x < 0) * 360
    return rdius, theta


def rotation_shape(coords_list, coords_origin, rotation_angle):
    rotation_coords_list = []
    for i in range(len(coords_list)):
        coords = coords_list[i]
        rdius, theta = polar360(coords[0], coords[1], coords_origin[0], coords_origin[1])
        x_r = np.int32(coords_origin[0] + rdius * math.sin((theta + rotation_angle) / 180 * math.pi))
        y_r = np.int32(coords_origin[1] + rdius * math.cos((theta + rotation_angle) / 180 * math.pi))
        rotation_coords_list.append([x_r, y_r])

    return rotation_coords_list


def drawimplant_coordinate(drawpaper_size, drawimplant_len, drawimplant_width):
    coord_center = drawpaper_size // 2
    len_center = drawimplant_len // 2
    width_center = drawimplant_width // 2
    coords_list = [[coord_center, coord_center], [coord_center + width_center, coord_center + len_center],
                   [coord_center + width_center, coord_center - len_center],
                   [coord_center - width_center, coord_center - len_center],
                   [coord_center - width_center, coord_center + len_center],
                   [coord_center, coord_center], [coord_center, drawpaper_size], [coord_center, 0]]

    return coords_list


def MaxMin_normalization_Intensity(I, Max_Minval, Min_Maxval):
    # ======================
    # I: HxW
    # ======================
    Ic = np.where(I > Min_Maxval, Min_Maxval, I)
    Ic = np.where(Ic < Max_Minval, Max_Minval, Ic)
    II = (Ic - Max_Minval) / (Min_Maxval - Max_Minval + 0.00001)

    return II


class Ui_MainWindow(object):
    def __init__(self):
        # 初始化其他属性...
        self.save_dicompath_temp = ""

    def setupUi(self, QMainWindow):
        self.QMainWindow = QMainWindow
        self.subject_name = 'Subject'
        self.threshold_ld = 0.5
        self.outputpath = './output/'
        # 设置模型参数
        # 添加变量
        os.environ['VIEWNIX_ENV'] = './CAVASS/'
        current_path = os.environ['Path']
        # 将新路径添加到PATH中，并移除重复项
        os.environ['Path'] = os.pathsep.join(current_path.split(os.pathsep) + ['./CAVASS/'])
        # ---------------------------------------
        self.args = Namespace()
        self.device = torch.device("cpu")
        self.args.image_size = 256
        self.args.encoder_adapter = True
        # self.args.sam_checkpoint = "./sam-med2d_refine.pth"
        # self.model = sam_model_registry["vit_b"](self.args).to(self.device)
        # ---------------植体放置2D图参数--------------------------------------------------------------------
        QMainWindow.setObjectName("MainWindow")
        QMainWindow.setWindowTitle('SAM-Med-Viewer')

        color = QtGui.QColor(255, 255, 255)  # RGB颜色，可以根据需要调整值
        QMainWindow.setStyleSheet(f"background-color: {color.name()};")
        QMainWindow.resize(1286, 1073)
        QMainWindow.setWindowIcon(QtGui.QIcon('./Tooth.ico'))

        self.centralwidget = QtWidgets.QWidget(QMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")

        self.font = QtGui.QFont()
        self.font.setFamily("华文宋体")
        self.font.setPointSize(10)

        self.font2 = QtGui.QFont()
        self.font2.setFamily("华文宋体")
        self.font2.setPointSize(11)
        # ------------------系统整体布局------------------------------
        self.system_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.system_layout.setSpacing(6)
        self.system_layout.setObjectName("system_layout")
        self.system_layout.addWidget(self.widget)
        # ----------------四视图整体布局-------------------------------
        self.four_view_layout = QtWidgets.QVBoxLayout()
        self.four_view_layout.setSpacing(6)
        self.four_view_layout.setObjectName("four_view_layout")

        self.xy_yz_horizontal_layout = QtWidgets.QHBoxLayout()
        self.xy_yz_horizontal_layout.setSpacing(6)
        self.xy_yz_horizontal_layout.setObjectName("xy_yz_horizontal_layout")
        # -------------------------XY 窗口----------------------------
        self.frame_XY = QtWidgets.QFrame(self.widget)
        self.frame_XY.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_XY.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_XY.setObjectName("frame_XY")
        self.id_XY = "XY"
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame_XY)
        self.xy_yz_horizontal_layout.addWidget(self.vtkWidget)
        # 打开横断面、矢状面、冠状面窗口数据
        self.pathDicomDir = getDirPath()
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.pathDicomDir)
        self.reader.Update()
        # 更新横断面
        self.viewer_XY = vtk.vtkImageViewer2()
        self.viewer_XY.SetInputData(self.reader.GetOutput())
        self.viewer_XY.SetupInteractor(self.vtkWidget)
        self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.viewer_XY.SetSliceOrientationToXY()
        self.viewer_XY.Render()
        # ------------------XY 滑块---------------------
        self.verticalSlider_XY = QtWidgets.QSlider(self.widget)
        self.verticalSlider_XY.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_XY.setObjectName("verticalSlider_XY")
        self.xy_yz_horizontal_layout.addWidget(self.verticalSlider_XY)
        # -------------------------YZ 窗口----------------------------
        self.frame_YZ = QtWidgets.QFrame(self.widget)
        self.frame_YZ.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_YZ.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_YZ.setObjectName("frame_YZ")
        self.id_YZ = "YZ"
        self.vtkWidget2 = QVTKRenderWindowInteractor(self.frame_YZ)
        self.xy_yz_horizontal_layout.addWidget(self.vtkWidget2)
        # 更新矢状面
        self.viewer_YZ = vtk.vtkImageViewer2()
        self.viewer_YZ.SetInputData(self.reader.GetOutput())
        self.viewer_YZ.SetupInteractor(self.vtkWidget2)
        self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
        self.viewer_YZ.SetSliceOrientationToYZ()
        self.viewer_YZ.Render()
        # ------------------XY 滑块---------------------
        self.verticalSlider_YZ = QtWidgets.QSlider(self.widget)
        self.verticalSlider_YZ.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_YZ.setObjectName("verticalSlider_YZ")
        self.xy_yz_horizontal_layout.addWidget(self.verticalSlider_YZ)
        # --------------------label------------------------、
        self.labelxy_labelyz_horizontal_layout = QtWidgets.QHBoxLayout()
        self.labelxy_labelyz_horizontal_layout.setSpacing(4)
        self.labelxy_labelyz_horizontal_layout.setObjectName("labelxy_labelyz_horizontal_layout")

        self.label_XY = QtWidgets.QLabel(self.widget)
        self.label_XY.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_XY.setObjectName("label_xy")

        self.label_YZ = QtWidgets.QLabel(self.widget)
        self.label_YZ.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_YZ.setObjectName("label_yz")

        self.labelxy_labelyz_horizontal_layout.addWidget(self.label_XY)
        self.labelxy_labelyz_horizontal_layout.addWidget(self.label_YZ)

        self.xy_yz_label_vertical_layout = QtWidgets.QVBoxLayout()
        self.xy_yz_label_vertical_layout.setSpacing(6)
        self.xy_yz_label_vertical_layout.setObjectName("xy_yz_label_vertical_layout")
        self.xy_yz_label_vertical_layout.addLayout(self.xy_yz_horizontal_layout)
        self.xy_yz_label_vertical_layout.addLayout(self.labelxy_labelyz_horizontal_layout)
        self.four_view_layout.addLayout(self.xy_yz_label_vertical_layout)

        # ---------------------------------------------------------------------------------------
        self.xz_volume_label_vertical_layout = QtWidgets.QVBoxLayout()
        self.xz_volume_label_vertical_layout.setSpacing(6)
        self.xz_volume_label_vertical_layout.setObjectName("xz_volume_label_vertical_layout")

        self.xz_volume_horizontal_layout = QtWidgets.QHBoxLayout()
        self.xz_volume_horizontal_layout.setSpacing(6)
        self.xz_volume_horizontal_layout.setObjectName("xz_volume_horizontal_layout")
        # -----------------------XZ窗口----------------------------------------------------------
        self.frame_XZ = QtWidgets.QFrame(self.widget)
        self.frame_XZ.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_XZ.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_XZ.setObjectName("frame_XZ")
        self.id_XZ = "XZ"
        self.vtkWidget3 = QVTKRenderWindowInteractor(self.frame_XZ)
        self.xz_volume_horizontal_layout.addWidget(self.vtkWidget3)
        # -------------------更新冠状面------------------------------------------------------------
        self.viewer_XZ = vtk.vtkImageViewer2()
        self.viewer_XZ.SetInputData(self.reader.GetOutput())
        self.viewer_XZ.SetupInteractor(self.vtkWidget3)
        self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
        self.viewer_XZ.SetSliceOrientationToYZ()
        self.viewer_XZ.Render()
        self.verticalSlider_XZ = QtWidgets.QSlider(self.widget)
        self.verticalSlider_XZ.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_XZ.setObjectName("verticalSlider_XZ")
        self.xz_volume_horizontal_layout.addWidget(self.verticalSlider_XZ)
        # ------------------------体绘制窗口-----------------------------------------------------------
        self.frame_Volume = QtWidgets.QFrame(self.widget)
        self.frame_Volume.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Volume.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Volume.setObjectName("frame_Volume")

        path = getDirPath()
        self.vtkWidget4, self.iren = volume(self, path)
        self.xz_volume_horizontal_layout.addWidget(self.vtkWidget4)

        self.verticalSlider_Volume = QtWidgets.QSlider(self.widget)
        self.verticalSlider_Volume.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_Volume.setObjectName("verticalSlider_Volume")
        self.xz_volume_horizontal_layout.addWidget(self.verticalSlider_Volume)
        self.xz_volume_horizontal_layout.setStretch(0, 10)
        self.xz_volume_horizontal_layout.setStretch(1, 1)
        self.xz_volume_horizontal_layout.setStretch(2, 10)
        self.xz_volume_horizontal_layout.setStretch(3, 1)
        self.xz_volume_label_vertical_layout.addLayout(self.xz_volume_horizontal_layout)
        # --------------------------主要是一些标签的设置------------------------------
        self.labelxz_labelvolume_horizontal_layout = QtWidgets.QHBoxLayout()
        self.labelxz_labelvolume_horizontal_layout.setSpacing(4)
        self.labelxz_labelvolume_horizontal_layout.setObjectName("labelxz_labelvolume_horizontal_layout")

        self.label_XZ = QtWidgets.QLabel(self.widget)
        self.label_XZ.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_XZ.setObjectName("label_XZ")

        self.label_Volume = QtWidgets.QLabel(self.widget)
        self.label_Volume.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Volume.setObjectName("label_Volume")

        self.labelxz_labelvolume_horizontal_layout.addWidget(self.label_XZ)
        self.labelxz_labelvolume_horizontal_layout.addWidget(self.label_Volume)

        self.xz_volume_label_vertical_layout.addLayout(self.labelxz_labelvolume_horizontal_layout)

        self.four_view_layout.addLayout(self.xz_volume_label_vertical_layout)
        self.system_layout.addLayout(self.four_view_layout, 7)

        # -----------------------------工具栏的布局-------------------------------------------------------------------
        self.tool_bar_layout = QtWidgets.QVBoxLayout()
        self.tool_bar_layout.setAlignment(Qt.AlignTop)
        self.tool_bar_layout.setSpacing(6)
        self.tool_bar_layout.setObjectName("tool_bar_layout")
        # -----------------------------对比度调整栏------------------------------------------------------------------
        self.widget_contrast = QtWidgets.QWidget(self.widget)
        self.widget_contrast.setMinimumSize(QtCore.QSize(350, 120))
        self.widget_contrast.setMaximumSize(QtCore.QSize(400, 120))
        self.widget_contrast.setObjectName("widget_contrast")
        self.widget_contrast.setStyleSheet('''background-color: #fafafa''')

        self.contrast_vertical_layout = QtWidgets.QVBoxLayout(self.widget_contrast)
        self.contrast_vertical_layout.setContentsMargins(11, 11, 11, 11)
        self.contrast_vertical_layout.setSpacing(5)
        self.contrast_vertical_layout.setObjectName("contrast_vertical_layout")
        # -----------------------------对比度调整栏名称---------------------------------------
        self.title = QtWidgets.QLabel(self.widget_contrast)
        self.title.setMinimumSize(QtCore.QSize(100, 20))
        self.title.setMaximumSize(QtCore.QSize(150, 20))
        self.title.setObjectName("title")
        self.title.setStyleSheet("color:green")
        self.title.setFont(self.font2)
        self.contrast_vertical_layout.addWidget(self.title, Qt.AlignLeft | Qt.AlignTop)
        # ----------------------------窗位线--------------------------------------------------
        self.window_level_slider = QtWidgets.QSlider(self.widget_contrast)
        self.window_level_slider.setOrientation(QtCore.Qt.Horizontal)
        self.window_level_slider.setObjectName("window_level_slider")
        self.window_level_slider.setMaximum(3000)
        self.window_level_slider.setMinimum(-2000)
        self.window_level_slider.setSingleStep(1)
        self.window_level_slider.valueChanged.connect(self.valuechange4)
        self.contrast_vertical_layout.addWidget(self.window_level_slider, Qt.AlignLeft)
        self.window_level = QtWidgets.QLabel(self.widget_contrast)
        self.window_level.setObjectName("window_level")
        self.contrast_vertical_layout.addWidget(self.window_level, Qt.AlignCenter)
        # ---------------------------窗宽线----------------------------------------------------
        self.window_width_slider = QtWidgets.QSlider(self.widget_contrast)
        self.window_width_slider.setOrientation(QtCore.Qt.Horizontal)
        self.window_width_slider.setObjectName("window_width_slider")
        self.window_width_slider.setMaximum(8000)
        self.window_width_slider.setMinimum(-2000)
        self.window_width_slider.setSingleStep(1)
        self.window_width_slider.valueChanged.connect(self.valuechange5)
        self.contrast_vertical_layout.addWidget(self.window_width_slider)
        self.window_width = QtWidgets.QLabel(self.widget_contrast)
        self.window_width.setObjectName("window_width")
        self.contrast_vertical_layout.addWidget(self.window_width, Qt.AlignCenter)
        self.tool_bar_layout.addWidget(self.widget_contrast)

        # Segmentation label button
        self.widget_labels = QtWidgets.QWidget(self.widget)
        self.widget_labels.setMinimumSize(QtCore.QSize(350, 100))
        self.widget_labels.setMaximumSize(QtCore.QSize(400, 100))
        self.widget_labels.hide()
        self.widget_labels.setObjectName("widget_label")
        self.widget_labels.setStyleSheet('''background-color: #fafafa''')
        self.labels_vertical_layout = QtWidgets.QVBoxLayout(self.widget_labels)
        self.labels_vertical_layout.setContentsMargins(11, 11, 11, 11)
        self.labels_vertical_layout.setSpacing(5)
        self.labels_vertical_layout.setAlignment(Qt.AlignTop)
        # -----------------------------title---------------------------------------
        self.widget_title = QtWidgets.QLabel(self.widget_labels)
        self.widget_title.setMinimumSize(QtCore.QSize(100, 20))
        self.widget_title.setMaximumSize(QtCore.QSize(150, 20))
        self.widget_title.setStyleSheet("color:green")
        self.widget_title.setFont(self.font2)
        self.labels_vertical_layout.addWidget(self.widget_title, Qt.AlignLeft | Qt.AlignTop)

        self.pushButton_clear = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_clear.setFont(self.font)
        self.pushButton_clear.setAutoExclusive(False)
        self.pushButton_clear.clicked.connect(self.label_clear)

        self.pushButton_undo = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_undo.setFont(self.font)
        self.pushButton_undo.setAutoExclusive(False)
        self.pushButton_undo.clicked.connect(self.label_undo)

        self.pushButton_redo = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_redo.setFont(self.font)
        self.pushButton_redo.setAutoExclusive(False)
        self.pushButton_redo.clicked.connect(self.label_redo)

        self.pushButton_layout = QtWidgets.QHBoxLayout()
        self.pushButton_layout.setSpacing(5)
        self.pushButton_layout.setContentsMargins(11, 11, 11, 11)
        self.pushButton_layout.addWidget(self.pushButton_redo)
        self.pushButton_layout.addWidget(self.pushButton_undo)
        self.pushButton_layout.addWidget(self.pushButton_clear)
        self.labels_vertical_layout.addLayout(self.pushButton_layout)
        self.tool_bar_layout.addWidget(self.widget_labels)

        self.system_layout.addLayout(self.tool_bar_layout, 2)
        # self.widget_registering.raise_()
        # self.resliceCursorRep_XY = vtk.vtkResliceCursorLineRepresentation()
        # self.resliceCursorRep_XZ = vtk.vtkResliceCursorLineRepresentation()
        # self.resliceCursorRep_YZ = vtk.vtkResliceCursorLineRepresentation()

        QMainWindow.setCentralWidget(self.centralwidget)
        # 主菜单栏样式
        menubar_style = """
                   QMenuBar{
                       background-color: rgba(255, 255, 255);
                       border: 1px solid rgba(240, 240, 240,);
                   }
                   QMenuBar::item {
                       color: rgb(0, 0, 0);
                       background: rgba(255, 255, 255);
                       padding: 4px 10px;
                   }
                   QMenuBar::item:selected {
                       background: rgba(48, 140, 1980);
                       color: rgb(255, 255, 255);
                   }
                   QMenuBar::item:pressed {
                       background: rgba(48, 140, 198,0.4);
                   }
               """
        self.menubar = QtWidgets.QMenuBar(QMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1103, 30))
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet(menubar_style)

        # 子菜单栏样式
        menu_style = """
                   QMenu {
                       background-color: rgba(255, 255, 255);
                       border: 1px solid rgba(244, 244, 244);
                   }
                   QMenu::item {
                       color: rgb(0, 0, 0);
                       background: rgba(255, 255, 255);
                   }
                   QMenu::item:selected {
                       background: rgba(48, 140, 198);
                       color: rgb(255, 255, 255);
                   }
                   QMenu::item:pressed {
                       background: rgba(48, 140, 198,0.4);
                   }
               """

        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setObjectName("fileMenu")
        self.fileMenu.setStyleSheet(menu_style)

        # self.toolMenu = QtWidgets.QMenu(self.menubar)
        # self.toolMenu.setObjectName("toolMenu")
        # self.toolMenu.setStyleSheet(menu_style)

        self.modelloadMenu = QtWidgets.QMenu(self.menubar)
        self.modelloadMenu.setObjectName("modelloadMenu")
        self.modelloadMenu.setStyleSheet(menu_style)

        self.segmentationMenu = QtWidgets.QMenu(self.menubar)
        self.segmentationMenu.setObjectName("segmentationMenu")
        self.segmentationMenu.setStyleSheet(menu_style)

        # 将主菜单栏添加到窗口中
        QMainWindow.setMenuBar(self.menubar)

        # 添加子菜单栏到主菜单栏中
        self.actionAdd_DiICOM_Data = QtWidgets.QAction(QMainWindow)
        self.actionAdd_DiICOM_Data.setObjectName("actionAdd_DiICOM_Data")
        self.actionAdd_DiICOM_Data.triggered.connect(self.on_actionAdd_DICOM_Data)

        self.actionAdd_NIFTI_Data = QtWidgets.QAction(QMainWindow)
        self.actionAdd_NIFTI_Data.triggered.connect(self.on_actionAdd_NIFTI_Data)

        self.actionAdd_IM0BIM_Data = QtWidgets.QAction(QMainWindow)
        self.actionAdd_IM0BIM_Data.setObjectName("actionAdd_IM0BIM_Data")
        self.actionAdd_IM0BIM_Data.triggered.connect(self.on_actionAdd_IM0BIM_Data)

        self.actionAdd_STL_Data = QtWidgets.QAction(QMainWindow)
        self.actionAdd_STL_Data.setObjectName("actionAdd_STL_Data")
        self.actionAdd_STL_Data.triggered.connect(self.on_actionAdd_STL_Data)
        self.reader_stl_renderer = vtk.vtkRenderer()
        self.reader_stl_renderer.SetBackground(0.5, 0.5, 0.5)
        self.reader_stl_renderer.ResetCamera()
        self.reader_stl_iren = self.vtkWidget4.GetRenderWindow().GetInteractor()
        self.vtkWidget4.GetRenderWindow().AddRenderer(self.reader_stl_renderer)
        self.reader_stl_style = vtk.vtkInteractorStyleTrackballCamera()
        self.reader_stl_style.SetDefaultRenderer(self.reader_stl_renderer)
        self.reader_stl_style.EnabledOn()

        # -----------------------------------------------------------------------
        self.actionAdd_Load_Universal_model = QtWidgets.QAction(QMainWindow)
        self.actionAdd_Load_Universal_model.setObjectName("actionAdd_Load_Universal_model")
        self.actionAdd_Load_Universal_model.triggered.connect(self.on_actionAdd_Load_Universal_model)

        self.actionAdd_Load_Lungseg_model = QtWidgets.QAction(QMainWindow)
        self.actionAdd_Load_Lungseg_model.setObjectName("actionAdd_Load_Lungseg_model")
        self.actionAdd_Load_Lungseg_model.triggered.connect(self.on_actionAdd_Load_Lungseg_model)

        self.Flag_Load_model = False
        # self.action_generatePanormaic = QtWidgets.QAction(QMainWindow)
        # self.action_generatePanormaic.setObjectName("action_generatePanormaic")
        # self.action_generatePanormaic.triggered.connect(self.on_action_generatePanormaic)
        #
        # self.action_implant_toolbar = QtWidgets.QAction(QMainWindow)
        # self.action_implant_toolbar.setObjectName("action_implant_toolbar")
        # self.action_implant_toolbar.setCheckable(True)
        # self.action_implant_toolbar.triggered.connect(self.on_action_implant_toolbar)
        #
        # self.action_registration_toolbar = QtWidgets.QAction(QMainWindow)
        # self.action_registration_toolbar.setObjectName("action_registration_toolbar")
        # self.action_registration_toolbar.setCheckable(True)
        # self.action_registration_toolbar.triggered.connect(self.on_action_registration_toolbar)
        #
        # self.action_parameters_toolbar = QtWidgets.QAction(QMainWindow)
        # self.action_parameters_toolbar.setObjectName("action_parameters_toolbar")
        # self.action_parameters_toolbar.setCheckable(True)
        # self.action_parameters_toolbar.triggered.connect(self.on_action_parameters_toolbar)

        # self.actionGroup = QtWidgets.QActionGroup(QMainWindow)
        # self.actionGroup.setExclusive(False)

        # 创建分割菜单栏中的操作
        self.pointAction = QtWidgets.QAction(QMainWindow)
        self.pointAction.setObjectName("pointAction")
        self.pointAction.setCheckable(True)
        self.pointAction.triggered.connect(self.on_action_point)
        # self.actionGroup.addAction(self.pointAction)

        self.point_label = QtWidgets.QMenu("Point Label", self.segmentationMenu)

        self.actionGroup = QtWidgets.QActionGroup(QMainWindow)
        # 添加子菜单项
        self.point_label_0 = QtWidgets.QAction("label 0", QMainWindow)
        self.point_label_0.setCheckable(True)
        self.point_label_0.setChecked(False)
        self.point_label_0.triggered.connect(self.select_point_label)
        self.actionGroup.addAction(self.point_label_0)

        self.point_label_1 = QtWidgets.QAction("label 1", QMainWindow)
        self.point_label_1.setCheckable(True)
        self.point_label_1.setChecked(True)
        self.point_label_1.triggered.connect(self.select_point_label)
        self.actionGroup.addAction(self.point_label_1)

        self.point_label.addAction(self.point_label_0)
        self.point_label.addAction(self.point_label_1)

        self.labelBoxAction = QtWidgets.QAction(QMainWindow)
        self.labelBoxAction.setObjectName("labelBoxAction")
        self.labelBoxAction.setCheckable(True)
        self.labelBoxAction.triggered.connect(self.on_action_labelBox)
        # self.actionGroup.addAction(self.labelBoxAction)

        self.box_label = QtWidgets.QMenu("Bounding-Box Type", self.segmentationMenu)
        self.boxlabel_actionGroup = QtWidgets.QActionGroup(QMainWindow)
        # 添加子菜单项
        self.box_label_single = QtWidgets.QAction("Single", QMainWindow)
        self.box_label_single.setCheckable(True)
        self.box_label_single.setObjectName("Single")
        self.box_label_single.setChecked(True)
        self.box_label_single.triggered.connect(self.select_box_label)
        self.boxlabel_actionGroup.addAction(self.box_label_single)

        self.box_label_multiple = QtWidgets.QAction("Multiple", QMainWindow)
        self.box_label_multiple.setObjectName("Multiple")
        self.box_label_multiple.setCheckable(True)
        self.box_label_multiple.setChecked(False)
        self.box_label_multiple.triggered.connect(self.select_box_label)
        self.boxlabel_actionGroup.addAction(self.box_label_multiple)

        self.box_label.addAction(self.box_label_single)
        self.box_label.addAction(self.box_label_multiple)

        self.segmentation_type = QtWidgets.QMenu("Segmentation Type", self.segmentationMenu)
        self.segmentation_type_group = QtWidgets.QActionGroup(QMainWindow)

        self.segmentation_type_none = QtWidgets.QAction("None", QMainWindow)
        self.segmentation_type_none.setCheckable(True)
        self.segmentation_type_none.setChecked(True)
        self.segmentation_type_none.setObjectName("None")
        self.segmentation_type_none.triggered.connect(self.select_slice_range)
        self.segmentation_type_group.addAction(self.segmentation_type_none)

        self.segmentation_type_sliceRange = QtWidgets.QAction("Slice Range", QMainWindow)
        self.segmentation_type_sliceRange.setCheckable(True)
        self.segmentation_type_sliceRange.setChecked(False)
        self.segmentation_type_sliceRange.setObjectName("Slice Range")
        self.segmentation_type_sliceRange.triggered.connect(self.select_slice_range)
        self.segmentation_type_group.addAction(self.segmentation_type_sliceRange)

        self.segmentation_type.addAction(self.segmentation_type_none)
        self.segmentation_type.addAction(self.segmentation_type_sliceRange)

        self.startSegmentationAction = QtWidgets.QAction(QMainWindow)
        self.startSegmentationAction.setObjectName("startSegmentationAction")
        self.startSegmentationAction.triggered.connect(self.on_action_startSegmentation)

        self.saveResultAction = QtWidgets.QAction(QMainWindow)
        self.saveResultAction.setObjectName("saveResultAction")
        self.saveResultAction.setCheckable(True)
        self.saveResultAction.triggered.connect(self.on_action_saveResult)

        self.fileMenu.addAction(self.actionAdd_DiICOM_Data)
        self.fileMenu.addAction(self.actionAdd_NIFTI_Data)
        self.fileMenu.addAction(self.actionAdd_IM0BIM_Data)
        self.fileMenu.addAction(self.actionAdd_STL_Data)

        self.modelloadMenu.addAction(self.actionAdd_Load_Universal_model)
        self.modelloadMenu.addAction(self.actionAdd_Load_Lungseg_model)

        self.segmentationMenu.addAction(self.pointAction)
        self.segmentationMenu.addMenu(self.point_label)
        self.segmentationMenu.addAction(self.labelBoxAction)
        self.segmentationMenu.addMenu(self.box_label)
        self.segmentationMenu.addAction(self.startSegmentationAction)
        self.segmentationMenu.addMenu(self.segmentation_type)
        self.segmentationMenu.addAction(self.saveResultAction)

        self.menubar.addAction(self.fileMenu.menuAction())
        self.menubar.addAction(self.modelloadMenu.menuAction())
        self.menubar.addAction(self.segmentationMenu.menuAction())

        # 窗口的状态栏 添加标签、进度条、临时信息等状态信息
        self.statusBar = QtWidgets.QStatusBar(QMainWindow)
        self.statusBar.setObjectName("statusBar")
        # 将状态栏添加到窗口中
        QMainWindow.setStatusBar(self.statusBar)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_dir = os.path.join(script_dir, 'img')
        # 为窗口设置工具栏
        self.toolBar = QtWidgets.QToolBar(QMainWindow)
        self.toolBar.setObjectName("toolBar")
        self.toolBar.setFixedHeight(40)

        # 在工具栏中添加各种功能
        QMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_ruler = QtWidgets.QAction(QMainWindow)
        self.action_ruler.setCheckable(True)
        self.action_ruler.setObjectName("action_ruler")
        self.action_ruler.setIcon(QtGui.QIcon(':/icons/img/ruler_logo.png'))
        ruler_icon_path = os.path.join(icon_dir, 'ruler_logo.png')

        # 调试输出
        print(f"尝试加载图标: {ruler_icon_path}")
        print(f"图标文件是否存在: {os.path.exists(ruler_icon_path)}")

        self.action_ruler.setIcon(QtGui.QIcon(ruler_icon_path))

        # 验证图标是否有效
        if self.action_ruler.icon().isNull():
            print("警告: 图标无效，可能路径错误或文件损坏")
            # 可以提供一个备选图标
            self.action_ruler.setIcon(QtGui.QIcon.fromTheme("help-contents"))
        self.action_ruler.triggered.connect(self.on_action_ruler)
        self.distance_widgets_1 = []  # 存储vtkDistanceWidget实例的列表
        self.distance_widgets_2 = []  # 存储vtkDistanceWidget实例的列表
        self.distance_widgets_3 = []  # 存储vtkDistanceWidget实例的列表
        self.ruler_enable = False

        self.action_paint = QtWidgets.QAction(QMainWindow)
        self.action_paint.setCheckable(True)
        self.action_paint.setObjectName("action_paint")
        self.action_paint.setIcon(QtGui.QIcon(':/icons/img/paint.png'))
        self.action_paint.triggered.connect(self.on_action_paint)
        self.paint_enable = False

        self.action_polyline = QtWidgets.QAction(QMainWindow)
        self.action_polyline.setCheckable(True)
        self.action_polyline.setObjectName("action_polyline")
        self.action_polyline.setIcon(QtGui.QIcon(':/icons/img/polyline.png'))
        self.action_polyline.triggered.connect(self.on_action_polyline)
        self.polyline_enable = False

        self.action_angle = QtWidgets.QAction(QMainWindow)
        self.action_angle.setCheckable(True)
        self.action_angle.setObjectName("action_angle")
        self.action_angle.setIcon(QtGui.QIcon(':/icons/img/angle.png'))
        self.action_angle.triggered.connect(self.on_action_angle)
        self.angle_enable = False

        self.action_pixel = QtWidgets.QAction(QMainWindow)
        self.action_pixel.setCheckable(True)
        self.action_pixel.setObjectName("action_pixel")
        self.action_pixel.setIcon(QtGui.QIcon(':/icons/img/view_value.png'))
        self.action_pixel.triggered.connect(self.on_action_pixel)
        self.pixel_enable = False

        self.action_crosshair = QtWidgets.QAction(QMainWindow)
        self.action_crosshair.setCheckable(True)
        self.action_crosshair.setObjectName("action_crosshair")
        self.action_crosshair.setIcon(QtGui.QIcon(":/icons/img/crosshair.png"))
        self.action_crosshair.triggered.connect(self.on_action_crosshair)
        self.gps_enable = False

        self.action_reset = QtWidgets.QAction(QMainWindow)
        self.action_reset.setObjectName("action_reset")
        self.action_reset.setIcon(QtGui.QIcon(':/icons/img/reset.png'))
        self.action_reset.triggered.connect(self.on_action_reset)

        # 拖动图像
        self.action_dragging_image = QtWidgets.QAction(QMainWindow)
        self.action_dragging_image.setObjectName("action_dragging_image")
        self.action_dragging_image.setCheckable(True)
        self.action_dragging_image.setIcon(QtGui.QIcon(':/icons/img/dragging.png'))
        self.action_dragging_image.triggered.connect(self.on_action_dragging_image)

        self.toolBar.addAction(self.action_ruler)
        self.toolBar.addAction(self.action_paint)
        self.toolBar.addAction(self.action_polyline)
        self.toolBar.addAction(self.action_angle)
        self.toolBar.addAction(self.action_pixel)
        self.toolBar.addAction(self.action_crosshair)
        self.toolBar.addAction(self.action_reset)
        self.toolBar.addAction(self.action_dragging_image)

        # 在工具栏的右侧添加间隔符
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolBar.addWidget(spacer)

        self.label_Subjectname = QtWidgets.QLabel(QMainWindow)
        self.label_Subjectname.setObjectName("label_Subjectname")
        self.label_Subjectname.setFixedHeight(25)
        self.label_Subjectname.setFixedWidth(110)
        self.label_Subjectname.setFont(self.font)
        self.label_Subjectname.setText("Subject Name: ")
        self.toolBar.addWidget(self.label_Subjectname)

        self.lineedit_Subjectname = QtWidgets.QLineEdit(QMainWindow)
        self.lineedit_Subjectname.setObjectName("lineedit_Subjectname")
        # self.lineedit_Subjectname.setMaximumSize(QtCore.QSize(300, 25))
        self.lineedit_Subjectname.setFixedHeight(25)
        self.lineedit_Subjectname.setFixedWidth(300)
        self.lineedit_Subjectname.setText('Subject')
        self.lineedit_Subjectname.textChanged[str].connect(self.lineedit_Subjectname_change_Func)  # 槽函数绑定
        self.toolBar.addWidget(self.lineedit_Subjectname)

        self.retranslateUi(QMainWindow)
        QtCore.QMetaObject.connectSlotsByName(QMainWindow)

    def retranslateUi(self, QMainWindow):
        _translate = QtCore.QCoreApplication.translate
        QMainWindow.setWindowTitle(_translate("MainWindow", "SAM-Med-Viewer"))
        self.fileMenu.setTitle(_translate("MainWindow", "File"))
        # self.toolMenu.setTitle(_translate("MainWindow", "Tools"))
        self.modelloadMenu.setTitle(_translate("MainWindow", "Load Models"))
        self.segmentationMenu.setTitle(_translate("MainWindow", "SAM-Med2D Seg"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionAdd_DiICOM_Data.setText(_translate("MainWindow", "Load DICOM"))
        self.actionAdd_NIFTI_Data.setText(_translate("MainWindow", "Load Seg"))
        self.actionAdd_IM0BIM_Data.setText(_translate("MainWindow", "Load IM0"))
        self.actionAdd_STL_Data.setText(_translate("MainWindow", "Load STL"))
        self.actionAdd_Load_Universal_model.setText(_translate("MainWindow", "Load Universal model"))
        self.actionAdd_Load_Lungseg_model.setText(_translate("MainWindow", "Load MRI Lungseg model"))
        self.action_ruler.setText(_translate("MainWindow", "Rule"))
        self.action_ruler.setToolTip(_translate("MainWindow",
                                                "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Ruler</span></p></body></html>"))
        self.action_paint.setText(_translate("MainWindow", "Paint"))
        self.action_paint.setToolTip(_translate("MainWindow",
                                                "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Paint</span></p></body></html>"))
        self.action_polyline.setText(_translate("MainWindow", "Line annotation"))
        self.action_polyline.setToolTip(_translate("MainWindow",
                                                   "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Line Annotation</span></p></body></html>"))
        self.action_angle.setText(_translate("MainWindow",
                                             "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Angle Measurement</span></p></body></html>"))
        self.action_pixel.setText(_translate("MainWindow",
                                             "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Density</span></p></body></html>"))
        self.action_reset.setText(_translate("MainWindow",
                                             "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Reset</span></p></body></html>"))
        self.action_crosshair.setText(_translate("MainWindow",
                                                 "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Synchronous Positioning</span></p></body></html>"))
        self.action_dragging_image.setText(_translate("MainWindow",
                                                      "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Drag</span></p></body></html>"))
        self.pointAction.setText(_translate("MainWindow", "Point"))
        self.labelBoxAction.setText(_translate("MainWindow", "Bounding-Box"))
        self.startSegmentationAction.setText(_translate("MainWindow", "Start Segmentation"))
        self.saveResultAction.setText(_translate("MainWindow", "Save Results"))
        self.widget_title.setText(_translate("QMainWindow", "Annotation"))
        self.pushButton_clear.setText(_translate("QMainWindow", "Clear Annotation"))
        self.pushButton_undo.setText(_translate("QMainWindow", "Withdraw"))
        self.pushButton_redo.setText(_translate("QMainWindow", "Recovery"))
        self.label_XY.setText(_translate("MainWindow", "Slice"))
        self.label_YZ.setText(_translate("MainWindow", "Slice"))
        self.label_XZ.setText(_translate("MainWindow", "Slice"))
        self.label_Volume.setText(_translate("MainWindow", "Volume"))
        self.title.setText(_translate("MainWindow", "Contrast Adjustment"))
        self.window_level.setText(_translate("MainWindow", "Window Level"))
        self.window_width.setText(_translate("MainWindow", "Window Width"))

    # 警告xinxi
    @staticmethod
    def message_dialog(title, text):
        msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title, text)
        msg_box.exec_()

    def lineedit_Subjectname_change_Func(self):
        self.subject_name = self.lineedit_Subjectname.text()
        print(self.subject_name)

    def implant_direction_cb_up_changed(self):
        if self.implant_direction_cb_up.isChecked():
            self.implant_direction_cb_down.setCheckState(0)

    def implant_direction_cb_down_changed(self):
        if self.implant_direction_cb_down.isChecked():
            self.implant_direction_cb_up.setCheckState(0)

    def anchor_direction_cb_up_changed(self):
        if self.anchor_direction_cb_up.isChecked():
            self.anchor_direction_cb_down.setCheckState(0)

    def anchor_direction_cb_down_changed(self):
        if self.anchor_direction_cb_down.isChecked():
            self.anchor_direction_cb_up.setCheckState(0)

    # ========================十字线定位========================================================================
    def on_action_crosshair(self):
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用十字定位功能")
            return
        if self.gps_enable == False:
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
            if self.pointAction.isChecked():
                self.label_clear()
                self.pointAction.setChecked(False)
                self.widget_labels.hide()
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.labelBoxAction.setChecked(False)
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                clearMultipleUndoStack()
                clearMultipleRedoStack()
            # --------------------------------XY-----------------------------------------------------------------
            flipYFilter = vtk.vtkImageFlip()
            flipYFilter.SetFilteredAxis(2)
            flipYFilter.FlipAboutOriginOff()
            flipYFilter.SetInputData(self.reader.GetOutput())
            flipYFilter.SetOutputSpacing(self.reader.GetDataSpacing())
            flipYFilter.Update()
            self.CT_Image = flipYFilter.GetOutput()
            # ------------------------------------------------------------
            self.renderWindowInteractor_XY = self.vtkWidget.GetRenderWindow().GetInteractor()
            self.interactorStyleImage_XY = vtk.vtkInteractorStyleImage()
            self.interactorStyleImage_XY.SetInteractor(self.renderWindowInteractor_XY)
            self.interactorStyleImage_XY.SetInteractionModeToImageSlicing()
            self.renderWindowInteractor_XY.SetInteractorStyle(self.interactorStyleImage_XY)
            self.viewer_XY = vtk.vtkResliceImageViewer()
            self.viewer_XY.SetInputData(self.CT_Image)
            self.viewer_XY.SetupInteractor(self.vtkWidget)
            self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
            self.viewer_XY.SetResliceModeToOblique()
            # -----------------------------------------------
            self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                0).SetRepresentationToWireframe()
            self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                1).SetRepresentationToWireframe()
            self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                2).SetRepresentationToWireframe()
            self.viewer_XY.GetResliceCursorWidget().SetManageWindowLevel(False)
            # --------------------------------------------------------------
            self.viewer_XY.SetSliceOrientationToXY()
            self.viewer_XY.Render()
            self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
            self.camera_XY.ParallelProjectionOn()  # 开启平行投影
            self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
            self.viewer_XY.SliceScrollOnMouseWheelOff()
            self.viewer_XY.Render()
            self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
            self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
            self.viewer_XY.GetResliceCursorWidget().AddObserver("MouseWheelForwardEvent", self.wheelforward1)
            self.viewer_XY.GetResliceCursorWidget().AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
            # --------------------------------YZ-----------------------------------------------------------------
            self.renderWindowInteractor_YZ = self.vtkWidget.GetRenderWindow().GetInteractor()
            self.interactorStyleImage_YZ = vtk.vtkInteractorStyleImage()
            self.interactorStyleImage_YZ.SetInteractor(self.renderWindowInteractor_YZ)
            self.interactorStyleImage_YZ.SetInteractionModeToImageSlicing()
            self.renderWindowInteractor_YZ.SetInteractorStyle(self.interactorStyleImage_YZ)
            self.viewer_YZ = vtk.vtkResliceImageViewer()
            self.viewer_YZ.SetInputData(self.CT_Image)
            self.viewer_YZ.SetupInteractor(self.vtkWidget2)
            self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
            self.viewer_YZ.SetResliceModeToOblique()
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(
                self.viewer_XY.GetResliceCursor())
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(
                0)
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                0).SetRepresentationToWireframe()
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                1).SetRepresentationToWireframe()
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                2).SetRepresentationToWireframe()
            self.viewer_YZ.GetResliceCursorWidget().SetManageWindowLevel(False)

            self.viewer_YZ.SetSliceOrientationToYZ()
            self.viewer_YZ.Render()
            self.viewer_YZ.SliceScrollOnMouseWheelOff()
            self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
            self.camera_YZ.ParallelProjectionOn()
            self.camera_YZ.SetParallelScale(80)
            self.viewer_YZ.Render()
            self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
            self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
            self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
            self.viewer_YZ.GetResliceCursorWidget().AddObserver("MouseWheelForwardEvent", self.wheelforward2)
            self.viewer_YZ.GetResliceCursorWidget().AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
            # --------------------------------XZ-----------------------------------------------------------------
            self.renderWindowInteractor_XZ = self.vtkWidget.GetRenderWindow().GetInteractor()
            self.interactorStyleImage_XZ = vtk.vtkInteractorStyleImage()
            self.interactorStyleImage_XZ.SetInteractor(self.renderWindowInteractor_XZ)
            self.interactorStyleImage_XZ.SetInteractionModeToImageSlicing()
            self.renderWindowInteractor_XZ.SetInteractorStyle(self.interactorStyleImage_XZ)
            self.viewer_XZ = vtk.vtkResliceImageViewer()

            self.viewer_XZ.SetInputData(self.CT_Image)
            self.viewer_XZ.SetupInteractor(self.vtkWidget3)
            self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
            self.viewer_XZ.SetResliceModeToOblique()
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(
                self.viewer_XY.GetResliceCursor())
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(
                1)
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                0).SetRepresentationToWireframe()
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                1).SetRepresentationToWireframe()
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().GetResliceCursorActor().GetCenterlineProperty(
                2).SetRepresentationToWireframe()
            self.viewer_XZ.GetResliceCursorWidget().SetManageWindowLevel(False)
            self.viewer_XZ.SetSliceOrientationToXZ()
            # -------------------------------------------------------
            self.viewer_XZ.Render()
            self.viewer_XZ.SliceScrollOnMouseWheelOff()
            self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
            self.camera_XZ.ParallelProjectionOn()
            self.camera_XZ.SetParallelScale(80)
            self.viewer_XZ.Render()
            # -------------------------------------------------------------------
            self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
            self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
            self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
            self.viewer_XZ.GetResliceCursorWidget().AddObserver("MouseWheelForwardEvent", self.wheelforward3)
            self.viewer_XZ.GetResliceCursorWidget().AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)
            # -------------------------------------------------------------------
            self.origin = self.reader.GetOutput().GetOrigin()
            self.spacing = self.reader.GetOutput().GetSpacing()
            center = self.reader.GetOutput().GetCenter()
            self.verticalSlider_XZ.setValue(int((center[0] - self.origin[0]) / self.spacing[0]))
            self.verticalSlider_YZ.setValue(int((center[1] - self.origin[1]) / self.spacing[1]))
            self.verticalSlider_XY.setValue(int((center[2] - self.origin[2]) / self.spacing[2]))
            self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
            self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))

            self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())

            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())

            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
            # -------------------------------------------------------------------------------------------------------------------
            self.commandSliceSelect_XY = CommandSelect('XY', self.reader, self.vtkWidget, self.vtkWidget2,
                                                       self.vtkWidget3, self.viewer_XY.GetResliceCursorWidget(),
                                                       self.viewer_YZ.GetResliceCursorWidget(),
                                                       self.viewer_XZ.GetResliceCursorWidget(),
                                                       self.viewer_XY.GetResliceCursor(), self.verticalSlider_XY,
                                                       self.verticalSlider_YZ, self.verticalSlider_XZ, self.label_XY,
                                                       self.label_YZ, self.label_XZ)
            self.viewer_XY.GetResliceCursorWidget().AddObserver(vtk.vtkResliceCursorWidget.ResliceAxesChangedEvent,
                                                                self.commandSliceSelect_XY)
            self.commandSliceSelect_YZ = CommandSelect('YZ', self.reader, self.vtkWidget, self.vtkWidget2,
                                                       self.vtkWidget3, self.viewer_XY.GetResliceCursorWidget(),
                                                       self.viewer_YZ.GetResliceCursorWidget(),
                                                       self.viewer_XZ.GetResliceCursorWidget(),
                                                       self.viewer_XY.GetResliceCursor(), self.verticalSlider_XY,
                                                       self.verticalSlider_YZ, self.verticalSlider_XZ, self.label_XY,
                                                       self.label_YZ, self.label_XZ)
            self.viewer_YZ.GetResliceCursorWidget().AddObserver(vtk.vtkResliceCursorWidget.ResliceAxesChangedEvent,
                                                                self.commandSliceSelect_YZ)
            self.commandSliceSelect_XZ = CommandSelect('XZ', self.reader, self.vtkWidget, self.vtkWidget2,
                                                       self.vtkWidget3, self.viewer_XY.GetResliceCursorWidget(),
                                                       self.viewer_YZ.GetResliceCursorWidget(),
                                                       self.viewer_XZ.GetResliceCursorWidget(),
                                                       self.viewer_XY.GetResliceCursor(), self.verticalSlider_XY,
                                                       self.verticalSlider_YZ, self.verticalSlider_XZ, self.label_XY,
                                                       self.label_YZ, self.label_XZ)
            self.viewer_XZ.GetResliceCursorWidget().AddObserver(vtk.vtkResliceCursorWidget.ResliceAxesChangedEvent,
                                                                self.commandSliceSelect_XZ)

            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()
            self.gps_enable = True
        else:
            # ------------------------------------------------------------------------
            self.viewer_XY.GetResliceCursorWidget().EnabledOff()
            self.viewer_YZ.GetResliceCursorWidget().EnabledOff()
            self.viewer_XZ.GetResliceCursorWidget().EnabledOff()

            self.window_level = self.window_level_slider.value()
            self.window_width = self.window_width_slider.value()
            print(self.window_level, self.window_width)
            self.pathDicomDir = getDirPath()
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.pathDicomDir)
            self.reader.Update()
            self.dims = self.reader.GetOutput().GetDimensions()
            self.dicomdata, self.header = load(self.pathDicomDir)
            # -------------------更新横断面------------------------------------------
            self.viewer_XY = vtk.vtkResliceImageViewer()
            self.viewer_XY.SetInputData(self.reader.GetOutput())
            self.viewer_XY.SetupInteractor(self.vtkWidget)
            self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
            self.viewer_XY.SetSliceOrientationToXY()
            value = self.verticalSlider_XY.value()
            self.viewer_XY.SetSlice(value)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
            self.camera_XY.ParallelProjectionOn()  # 开启平行投影
            self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
            self.viewer_XY.SliceScrollOnMouseWheelOff()
            self.viewer_XY.Render()
            # --------------------------------------------------------------------------------------
            self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
            self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
            self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
            # -------------------更新矢状面------------------------------------------------------------
            self.viewer_YZ = vtk.vtkResliceImageViewer()
            self.viewer_YZ.SetInputData(self.reader.GetOutput())
            self.viewer_YZ.SetupInteractor(self.vtkWidget2)
            self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
            self.viewer_YZ.SetSliceOrientationToYZ()
            value = self.verticalSlider_YZ.value()
            self.viewer_YZ.SetSlice(value)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            self.viewer_YZ.SliceScrollOnMouseWheelOff()
            self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
            self.camera_YZ.ParallelProjectionOn()
            self.camera_YZ.SetParallelScale(80)
            # ---------------------------------------------------------------------------------------
            bounds = self.reader.GetOutput().GetBounds()
            center0 = (bounds[1] + bounds[0]) / 2.0
            center1 = (bounds[3] + bounds[2]) / 2.0
            center2 = (bounds[5] + bounds[4]) / 2.0

            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)
            transform_YZ.RotateX(180)
            transform_YZ.RotateZ(180)
            transform_YZ.Translate(-center0, -center1, -center2)
            self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
            # ----------------------------------------------------------------------------------------
            self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
            self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
            self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
            self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
            self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
            # -------------更新冠状面---------------------------------------------------------------
            self.viewer_XZ = vtk.vtkResliceImageViewer()
            self.viewer_XZ.SetInputData(self.reader.GetOutput())
            self.viewer_XZ.SetupInteractor(self.vtkWidget3)
            self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
            self.viewer_XZ.SetSliceOrientationToXZ()
            value = self.verticalSlider_XZ.value()
            self.viewer_XZ.SetSlice(value)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            self.viewer_XZ.SliceScrollOnMouseWheelOff()
            self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
            self.camera_XZ.ParallelProjectionOn()
            self.camera_XZ.SetParallelScale(80)
            # -----------------------------------------------------------------------------------
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)
            transform_XZ.RotateZ(180)
            transform_XZ.Translate(-center0, -center1, -center2)
            self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
            # --------------------------------------------------------------------------------------
            self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
            self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
            self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
            self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)

            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()

            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()
            self.gps_enable = False

    def cross_hairaxis_orthogonal(self):
        if self.cross_hairaxis_orthogonal_enable == False:
            self.cross_hairaxis_orthogonal_enable = True
            pyautogui.keyDown('ctrl')  # 保持ctrl键取消
        else:
            self.cross_hairaxis_orthogonal_enable = False
            pyautogui.keyUp('ctrl')  # 保持ctrl键取消

    # -------------------------------Update--------------------------------
    def valuechange(self):
        if self.gps_enable == False:
            if self.pointAction.isChecked():
                try:
                    for i in getPointsActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                value = self.verticalSlider_XY.value()
                self.viewer_XY.SetSlice(value)
                if getPointsUndoStack() != []:
                    for point in getPointsUndoStack():
                        if point[2] == value:
                            print("valuechange")
                            self.point_paints(point)
            if self.labelBoxAction.isChecked():
                if getSelectSingleBoxLabel():
                    try:
                        for i in getSingleBoundingBoxActor():
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                        self.viewer_XY.Render()
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    value = self.verticalSlider_XY.value()
                    self.viewer_XY.SetSlice(value)
                    if getSingleUndoStack() != []:
                        for data in getSingleUndoStack():
                            if data[4] == self.verticalSlider_XY.value():
                                print("single redo")
                                self.actor_list = []
                                self.drwa_single_bounding_box(data)
                                setSingleBoundingBoxActor(self.actor_list)
                    self.viewer_XY.UpdateDisplayExtent()
                    self.viewer_XY.Render()
                else:
                    try:
                        for i in getSingleBoundingBoxActor():
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    try:
                        for i in getLastBoundingBoxActor():
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    try:
                        for actor in getMultipleBoundingBoxActor():
                            for i in actor:
                                self.viewer_XY.GetRenderer().RemoveActor(i)
                        clearMultipleBoundingBoxActor()
                        self.viewer_XY.Render()
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    value = self.verticalSlider_XY.value()
                    self.viewer_XY.SetSlice(value)
                    if getMultipleUndoStack() != []:
                        for data in getMultipleUndoStack():
                            if data[4] == self.verticalSlider_XY.value():
                                print("multiple redo")
                                self.actor_list = []
                                self.drwa_single_bounding_box(data)
                                setMultipleBoundingBoxActor(self.actor_list)
                    self.viewer_XY.UpdateDisplayExtent()
                    self.viewer_XY.Render()
            else:
                print("ResliceImageView Change")
                value = self.verticalSlider_XY.value()
                self.viewer_XY.SetSlice(value)

            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            try:
                self.viewer_seg_xy.SetSlice(value)
                self.viewer_seg_xy.UpdateDisplayExtent()
                self.viewer_seg_xy.Render()
                self.viewer_dicom_xy.SetSlice(value)
                self.viewer_dicom_xy.UpdateDisplayExtent()
                self.viewer_dicom_xy.Render()
            except:
                print("Don't exist")
            self.label_XY.setText("Slice %d/%d" % (self.viewer_XY.GetSlice(), self.viewer_XY.GetSliceMax()))
        else:
            center = list(self.viewer_XY.GetResliceCursor().GetCenter())
            value_XY = self.verticalSlider_XY.value()
            value_YZ = self.verticalSlider_YZ.value()
            value_XZ = self.verticalSlider_XZ.value()
            origin = self.reader.GetOutput().GetOrigin()
            spacing = self.reader.GetOutput().GetSpacing()
            # center_X = value_YZ * spacing[0] - origin[0]
            # center_Y = value_XZ * spacing[1] - origin[1]
            center_Z = value_XY * spacing[2] - origin[2]
            # 更新 Z
            center[2] = center_Z
            self.viewer_XY.GetResliceCursor().SetCenter(center)
            self.viewer_XY.GetResliceCursor().Update()
            self.viewer_XY.GetResliceCursorWidget().Render()
            self.viewer_YZ.GetResliceCursorWidget().Render()
            self.viewer_XZ.GetResliceCursorWidget().Render()
            self.label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # self.resliceCursorRep_XY.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.resliceCursor)
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()

    def valuechange2(self):
        if self.gps_enable == False:
            value = self.verticalSlider_YZ.value()
            self.viewer_YZ.SetSlice(value)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            try:
                self.viewer_seg_yz.SetSlice(value)
                self.viewer_seg_yz.UpdateDisplayExtent()
                self.viewer_seg_yz.Render()
                self.viewer_dicom_yz.SetSlice(value)
                self.viewer_dicom_yz.UpdateDisplayExtent()
                self.viewer_dicom_yz.Render()
            except:
                print("Don't exist")
            self.label_YZ.setText("Slice %d/%d" % (self.viewer_YZ.GetSlice(), self.viewer_YZ.GetSliceMax()))
        else:
            center = list(self.viewer_XY.GetResliceCursor().GetCenter())
            value_XY = self.verticalSlider_XY.value()
            value_YZ = self.verticalSlider_YZ.value()
            value_XZ = self.verticalSlider_XZ.value()
            origin = self.reader.GetOutput().GetOrigin()
            spacing = self.reader.GetOutput().GetSpacing()
            center_X = value_YZ * spacing[0] - origin[0]
            # center_Y = value_XZ * spacing[1] - origin[1]
            # center_Z = value_XY * spacing[2] - origin[2]
            center[0] = center_X
            self.viewer_XY.GetResliceCursor().SetCenter(center)
            self.viewer_XY.GetResliceCursor().Update()
            self.viewer_XY.GetResliceCursorWidget().Render()
            self.viewer_YZ.GetResliceCursorWidget().Render()
            self.viewer_XZ.GetResliceCursorWidget().Render()

            self.label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # self.resliceCursorRep_XY.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.resliceCursor)
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()

    def valuechange3(self):
        if self.gps_enable == False:
            value = self.verticalSlider_XZ.value()
            self.viewer_XZ.SetSlice(value)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            try:
                self.viewer_seg_xz.SetSlice(value)
                self.viewer_seg_xz.UpdateDisplayExtent()
                self.viewer_seg_xz.Render()
                self.viewer_dicom_xz.SetSlice(value)
                self.viewer_dicom_xz.UpdateDisplayExtent()
                self.viewer_dicom_xz.Render()
            except:
                print("Don't exist")
            self.label_XZ.setText("Slice %d/%d" % (self.viewer_XZ.GetSlice(), self.viewer_XZ.GetSliceMax()))
        else:
            center = list(self.viewer_XY.GetResliceCursor().GetCenter())
            value_XY = self.verticalSlider_XY.value()
            value_YZ = self.verticalSlider_YZ.value()
            value_XZ = self.verticalSlider_XZ.value()
            origin = self.reader.GetOutput().GetOrigin()
            spacing = self.reader.GetOutput().GetSpacing()
            # center_X = value_YZ * spacing[0] - origin[0]
            center_Y = value_XZ * spacing[1] - origin[1]
            # center_Z = value_XY * spacing[2] - origin[2]
            center[1] = center_Y
            self.viewer_XY.GetResliceCursor().SetCenter(center)
            self.viewer_XY.GetResliceCursor().Update()
            self.viewer_XY.GetResliceCursorWidget().Render()
            self.viewer_YZ.GetResliceCursorWidget().Render()
            self.viewer_XZ.GetResliceCursorWidget().Render()
            self.label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # self.resliceCursorRep_XY.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.resliceCursor)
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()

    def valuechange4(self):
        if getFileIsEmpty() == True:
            print("未导入文件，不能修改窗位数值")
            return

        self.window_level_slider.setToolTip(str(self.window_level_slider.value()))
        self.window_width_slider.setToolTip(str(self.window_width_slider.value()))
        self.level = self.viewer_XY.GetColorLevel()
        self.wi = self.viewer_XY.GetColorWindow()
        try:
            self.viewer_dicom_xy.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_xy.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_yz.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_yz.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_xz.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_xz.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_xy.Render()
            self.viewer_dicom_yz.Render()
            self.viewer_dicom_xz.Render()
        except:
            print("Don't exist")
        self.viewer_XY.SetColorLevel(self.window_level_slider.value())
        self.viewer_XY.SetColorWindow(self.window_width_slider.value())
        self.viewer_XY.Render()
        self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
        self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
        self.viewer_YZ.Render()
        self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
        self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
        self.viewer_XZ.Render()
        self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
            self.window_width_slider.value(), self.window_level_slider.value())
        self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
            self.window_width_slider.value(), self.window_level_slider.value())
        self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
            self.window_width_slider.value(), self.window_level_slider.value())

    def valuechange5(self):
        if getFileIsEmpty() == True:
            print("未导入文件，不能修改窗宽数值")
            return

        self.window_level_slider.setToolTip(str(self.window_level_slider.value()))
        self.window_width_slider.setToolTip(str(self.window_width_slider.value()))
        try:
            self.viewer_dicom_xy.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_xy.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_yz.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_yz.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_xz.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_xz.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_xy.Render()
            self.viewer_dicom_yz.Render()
            self.viewer_dicom_xz.Render()
        except:
            print("Don't exist")
        self.viewer_XY.SetColorLevel(self.window_level_slider.value())
        self.viewer_XY.SetColorWindow(self.window_width_slider.value())
        self.viewer_XY.Render()
        self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
        self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
        self.viewer_YZ.Render()
        self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
        self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
        self.viewer_XZ.Render()
        self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
            self.window_width_slider.value(), self.window_level_slider.value())
        self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
            self.window_width_slider.value(), self.window_level_slider.value())
        self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
            self.window_width_slider.value(), self.window_level_slider.value())

    def switch(self):
        if getNumber() % 2 == 0:
            self.state = False
        else:
            self.state = True
        setNumber()

    def on_actionAdd_Load_Universal_model(self):
        print('Load Universal_model!')
        # path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "", "*.pth;")
        # -----------------------------------------------------
        if self.Flag_Load_model == True:
            del self.model
        else:
            self.Flag_Load_model = True
        # -----------------------------------------------------
        self.args.sam_checkpoint = "./sam-med2d_refine.pth"
        self.model = sam_model_registry["vit_b"](self.args).to(self.device)
        self.message_dialog('Load model', 'Load Universal_model Successfully!')
        self.modeltype = 'Universal'
        print('Load Universal_model Successfully!')

    def on_actionAdd_Load_Lungseg_model(self):
        print("Load Lungseg_model!")
        try:
            # 直接使用 None 作为父窗口
            model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                None,  # 关键修改：父窗口设为 None
                "选择Lungseg模型文件",
                "./",
                "模型文件 (*.pth);;所有文件 (*)"
            )

            if not model_path:
                QtWidgets.QMessageBox.information(
                    None,  # 父窗口设为 None
                    "提示",
                    "未选择模型文件，加载取消。"
                )
                return

            if not os.path.exists(model_path):
                QtWidgets.QMessageBox.critical(
                    parent_window,
                    "错误",
                    f"模型文件不存在：{model_path}"
                )
                return

            # 加载模型
            self.args.checkpoint = model_path
            self.model = sam_model_registry["vit_b"](self.args).to(self.device)
            QtWidgets.QMessageBox.information(
                parent_window,
                "成功",
                f"模型已加载：{model_path}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,  # 父窗口设为 None
                "加载失败",
                f"模型加载出错：{str(e)}"
            )

    def on_actionAdd_NIFTI_Data(self):

        path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "", "*.nii.gz")
        if path[0] == "":
            return

        self.imageblend_seg_mask(path[0])

    def imageblend_seg_mask(self, path, slice_index=None):
        self.reader_seg = vtk.vtkNIFTIImageReader()
        self.reader_seg.SetFileName(path)
        self.reader_seg.Update()

        dicom_image = self.reader.GetOutput()
        seg_image = self.reader_seg.GetOutput()

        flip1 = vtk.vtkImageFlip()
        flip1.SetInputData(seg_image)
        flip1.SetFilteredAxis(2)
        flip1.Update()
        flip2 = vtk.vtkImageFlip()
        flip2.SetInputData(flip1.GetOutput())
        flip2.SetFilteredAxis(1)
        flip2.Update()
        # 确保分割结果与DICOM的几何信息一致
        change_info = vtk.vtkImageChangeInformation()
        change_info.SetInputConnection(flip2.GetOutputPort())
        change_info.SetOutputOrigin(dicom_image.GetOrigin())
        change_info.SetOutputSpacing(dicom_image.GetSpacing())
        change_info.Update()
        self.seg_image_aligned = change_info.GetOutput()

        # 获取分割图像的实际数据范围
        data_min, data_max = self.seg_image_aligned.GetScalarRange()

        self.color_table = vtk.vtkLookupTable()
        self.color_table.SetNumberOfColors(256)  # 扩展颜色精度
        self.color_table.SetTableRange(data_min, data_max)  # 动态适配
        self.color_table.SetTableValue(0, 0.0, 0.0, 1.0, 0.0)  # 背景透明
        for i in range(1, 256):
            self.color_table.SetTableValue(i, 1, 0, 0, 1.0)  # 红色不透明
        self.color_table.Build()

        self.viewer_dicom_xy, self.viewer_xy_camera = self.create_dicom_viewer(self.vtkWidget.GetRenderWindow(), "XY",
                                                                               slice_index)
        self.viewer_dicom_yz, self.viewer_yz_camera = self.create_dicom_viewer(self.vtkWidget2.GetRenderWindow(), "YZ",
                                                                               slice_index)
        self.viewer_dicom_xz, self.viewer_xz_camera = self.create_dicom_viewer(self.vtkWidget3.GetRenderWindow(), "XZ",
                                                                               slice_index)

        self.viewer_camera_XY_focalPoint = self.viewer_xy_camera.GetFocalPoint()
        self.viewer_camera_XY_position = self.viewer_xy_camera.GetPosition()
        self.viewer_camera_YZ_focalPoint = self.viewer_yz_camera.GetFocalPoint()
        self.viewer_camera_YZ_position = self.viewer_yz_camera.GetPosition()
        self.viewer_camera_XZ_focalPoint = self.viewer_xz_camera.GetFocalPoint()
        self.viewer_camera_XZ_position = self.viewer_xz_camera.GetPosition()

        self.viewer_seg_xy, self.viewer_dicom_interactor_xy = self.create_seg_viewer(self.vtkWidget.GetRenderWindow(),
                                                                                     "XY", self.viewer_dicom_xy,
                                                                                     slice_index)
        self.viewer_seg_yz, self.viewer_dicom_interactor_yz = self.create_seg_viewer(self.vtkWidget2.GetRenderWindow(),
                                                                                     "YZ", self.viewer_dicom_yz,
                                                                                     slice_index)
        self.viewer_seg_xz, self.viewer_dicom_interactor_xz = self.create_seg_viewer(self.vtkWidget3.GetRenderWindow(),
                                                                                     "XZ", self.viewer_dicom_xz,
                                                                                     slice_index)

        self.viewer_dicom_xy.Render()
        self.viewer_dicom_yz.Render()
        self.viewer_dicom_xz.Render()
        self.viewer_seg_xy.Render()
        self.viewer_seg_yz.Render()
        self.viewer_seg_xz.Render()

    def create_dicom_viewer(self, render_window, viewer_type, slice_index):
        bounds = self.reader.GetOutput().GetBounds()
        center0 = (bounds[1] + bounds[0]) / 2.0
        center1 = (bounds[3] + bounds[2]) / 2.0
        center2 = (bounds[5] + bounds[4]) / 2.0
        viewer_dicom = vtk.vtkImageViewer2()
        viewer_dicom.SetInputData(self.reader.GetOutput())
        viewer_dicom.SetRenderWindow(render_window)
        viewer_dicom.UpdateDisplayExtent()
        if viewer_type == "XY":
            if slice_index == None:
                viewer_dicom.SetSlice(int(self.viewer_XY.GetSliceMax() / 2))
                self.verticalSlider_XY.setValue(int(viewer_dicom.GetSliceMax() / 2))
                self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), viewer_dicom.GetSliceMax()))
            else:
                viewer_dicom.SetSlice(slice_index)
                self.verticalSlider_XY.setValue(slice_index)
                self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), viewer_dicom.GetSliceMax()))

            viewer_dicom.SetSliceOrientationToXY()
        if viewer_type == "YZ":
            viewer_dicom.SetSlice(int(self.viewer_YZ.GetSliceMax() / 2))
            self.verticalSlider_YZ.setValue(int(viewer_dicom.GetSliceMax() / 2))
            self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), viewer_dicom.GetSliceMax()))
            viewer_dicom.SetSliceOrientationToYZ()
            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)
            transform_YZ.RotateX(180)
            transform_YZ.RotateZ(180)
            transform_YZ.Translate(-center0, -center1, -center2)
            viewer_dicom.GetImageActor().SetUserTransform(transform_YZ)
        if viewer_type == "XZ":
            viewer_dicom.SetSlice(int(self.viewer_XZ.GetSliceMax() / 2))
            self.verticalSlider_XZ.setValue(int(viewer_dicom.GetSliceMax() / 2))
            self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), viewer_dicom.GetSliceMax()))
            viewer_dicom.SetSliceOrientationToXZ()
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)
            transform_XZ.RotateZ(180)
            transform_XZ.Translate(-center0, -center1, -center2)
            viewer_dicom.GetImageActor().SetUserTransform(transform_XZ)
        viewer_dicom.SetColorWindow(self.window_width_slider.value())
        viewer_dicom.SetColorLevel(self.window_level_slider.value())
        camera = viewer_dicom.GetRenderer().GetActiveCamera()
        camera.ParallelProjectionOn()  # 开启平行投影
        camera.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
        return viewer_dicom, camera

    def create_seg_viewer(self, render_window, viewer_type, viewer_dicom, slice_index):
        bounds = self.reader.GetOutput().GetBounds()
        center0 = (bounds[1] + bounds[0]) / 2.0
        center1 = (bounds[3] + bounds[2]) / 2.0
        center2 = (bounds[5] + bounds[4]) / 2.0
        viewerLayer = vtk.vtkImageViewer2()
        viewerLayer.SetInputData(self.seg_image_aligned)
        viewerLayer.SetRenderWindow(render_window)
        if viewer_type == "XY":
            viewerLayer.SetSliceOrientationToXY()
            rwi = self.viewer_XY.GetRenderWindow().GetInteractor()
            wheelforward = MouseWheelForward(viewer_dicom, self.label_XY, self.verticalSlider_XY,
                                             self.id_XY)
            wheelbackward = MouseWheelBackWard(viewer_dicom, self.label_XY, self.verticalSlider_XY,
                                               self.id_XY)
            if slice_index == None:
                viewerLayer.SetSlice(viewer_dicom.GetSlice())
            else:
                viewerLayer.SetSlice(slice_index)
        elif viewer_type == "YZ":
            viewerLayer.SetSliceOrientationToYZ()
            rwi = self.viewer_YZ.GetRenderWindow().GetInteractor()
            wheelforward = MouseWheelForward(viewer_dicom, self.label_YZ, self.verticalSlider_YZ,
                                             self.id_YZ)
            wheelbackward = MouseWheelBackWard(viewer_dicom, self.label_YZ, self.verticalSlider_YZ,
                                               self.id_YZ)
            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)
            transform_YZ.RotateX(180)
            transform_YZ.RotateZ(180)
            transform_YZ.Translate(-center0, -center1, -center2)
            viewerLayer.GetImageActor().SetUserTransform(transform_YZ)
            viewerLayer.SetSlice(viewer_dicom.GetSlice())
        else:
            viewerLayer.SetSliceOrientationToXZ()
            rwi = self.viewer_XZ.GetRenderWindow().GetInteractor()
            wheelforward = MouseWheelForward(viewer_dicom, self.label_XZ, self.verticalSlider_XZ,
                                             self.id_XZ)
            wheelbackward = MouseWheelBackWard(viewer_dicom, self.label_XZ, self.verticalSlider_XZ,
                                               self.id_XZ)
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)
            transform_XZ.RotateZ(180)
            transform_XZ.Translate(-center0, -center1, -center2)
            viewerLayer.GetImageActor().SetUserTransform(transform_XZ)
            viewerLayer.SetSlice(viewer_dicom.GetSlice())
        viewerLayer.GetImageActor().SetInterpolate(False)
        viewerLayer.GetImageActor().GetProperty().SetLookupTable(self.color_table)
        viewerLayer.GetImageActor().GetProperty().SetDiffuse(0.0)
        viewerLayer.GetImageActor().SetPickable(False)
        viewer_dicom.GetRenderer().AddActor(viewerLayer.GetImageActor())
        viewer_dicom.SetupInteractor(rwi)
        viewer_dicom_interactor = viewer_dicom.GetInteractorStyle()
        # --------------------------------------------------------------------------------------
        viewer_dicom_interactor.AddObserver("MouseWheelForwardEvent", wheelforward)
        viewer_dicom_interactor.AddObserver("MouseWheelBackwardEvent", wheelbackward)
        return viewerLayer, viewer_dicom_interactor

    def on_actionAdd_DICOM_Data(self):
        self.dataformat = 'DICOM'
        print("选择DICOM文件")
        old_path = getDirPath()
        path = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹")

        if path == "":
            if old_path == 'F:\CBCT_Register_version_12_7\testdata\\40':
                setFileIsEmpty(True)
                return
            else:
                path = old_path

        self.action_ruler.setChecked(False)
        self.action_paint.setChecked(False)
        self.action_pixel.setChecked(False)
        self.action_polyline.setChecked(False)
        self.action_angle.setChecked(False)
        self.action_crosshair.setChecked(False)
        self.action_dragging_image.setChecked(False)
        self.pointAction.setChecked(False)
        self.labelBoxAction.setChecked(False)
        self.saveResultAction.setChecked(False)
        self.toolBar.update()
        # 获取目录下的所有文件名
        files = os.listdir(path)
        # 检查是否存在DCM文件
        dcm_files_exist = any(file.endswith(".dcm") for file in files)

        if not dcm_files_exist:
            print("该目录下没有DCM文件数据")
            return
        # 清除之前的直线
        if self.ruler_enable == True:
            for ruler1 in self.distance_widgets_1:
                ruler1.Off()
            self.distance_widgets_1.clear()  # 清空列表
            for ruler2 in self.distance_widgets_2:
                ruler2.Off()
            self.distance_widgets_2.clear()  # 清空列表
            for ruler3 in self.distance_widgets_3:
                ruler3.Off()
            self.distance_widgets_3.clear()  # 清空列表
            self.ruler_enable = False
        # 清除之前的角度测量
        if self.angle_enable == True:
            self.angleWidget1.Off()
            self.angleWidget2.Off()
            self.angleWidget3.Off()
            self.angle_enable = False
        if self.paint_enable == True:
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            try:
                for i in getActors_paint():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_YZ.GetRenderer().RemoveActor(i)
                self.viewer_YZ.Render()
            except:
                print('Close viewer_YZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_XZ.GetRenderer().RemoveActor(i)
                self.viewer_XZ.Render()
            except:
                print('Close viewer_XZ actor_paint Failed!!!')
            self.paint_enable = False
        if self.polyline_enable == True:
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            # ---------------------------------------------------------------
            try:
                for i in getActors_paint():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_YZ.GetRenderer().RemoveActor(i)
                self.viewer_YZ.Render()
            except:
                print('Close viewer_YZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_XZ.GetRenderer().RemoveActor(i)
                self.viewer_XZ.Render()
            except:
                print('Close viewer_XZ actor_paint Failed!!!')
            self.polyline_enable = False
        if self.pixel_enable == True:
            self.vtkWidget.setToolTip('')
            self.vtkWidget2.setToolTip('')
            self.vtkWidget3.setToolTip('')
            self.imagestyle1.RemoveObservers("MouseMoveEvent")
            self.imagestyle2.RemoveObservers("MouseMoveEvent")
            self.imagestyle3.RemoveObservers("MouseMoveEvent")
            self.pixel_enable = False
        if self.gps_enable == True:
            self.action_crosshair.setChecked(False)
            # ------------------------------------------------------------------------
            self.viewer_XY.GetResliceCursorWidget().EnabledOff()
            self.viewer_YZ.GetResliceCursorWidget().EnabledOff()
            self.viewer_XZ.GetResliceCursorWidget().EnabledOff()

            self.window_level = self.window_level_slider.value()
            self.window_width = self.window_width_slider.value()
            print(self.window_level, self.window_width)
            self.pathDicomDir = getDirPath()
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.pathDicomDir)
            self.reader.Update()
            self.dims = self.reader.GetOutput().GetDimensions()
            self.dicomdata, self.header = load(self.pathDicomDir)
            # -------------------更新横断面------------------------------------------
            self.viewer_XY = vtk.vtkResliceImageViewer()
            self.viewer_XY.SetInputData(self.reader.GetOutput())
            self.viewer_XY.SetupInteractor(self.vtkWidget)
            self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
            self.viewer_XY.SetSliceOrientationToXY()
            value = self.verticalSlider_XY.value()
            self.viewer_XY.SetSlice(value)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
            self.camera_XY.ParallelProjectionOn()  # 开启平行投影
            self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
            self.viewer_XY.SliceScrollOnMouseWheelOff()
            self.viewer_XY.Render()
            # --------------------------------------------------------------------------------------
            self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                   self.id_XY)
            self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                     self.id_XY)
            self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
            # -------------------更新矢状面------------------------------------------------------------
            self.viewer_YZ = vtk.vtkResliceImageViewer()
            self.viewer_YZ.SetInputData(self.reader.GetOutput())
            self.viewer_YZ.SetupInteractor(self.vtkWidget2)
            self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
            self.viewer_YZ.SetSliceOrientationToYZ()
            value = self.verticalSlider_YZ.value()
            self.viewer_YZ.SetSlice(value)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            self.viewer_YZ.SliceScrollOnMouseWheelOff()
            self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
            self.camera_YZ.ParallelProjectionOn()
            self.camera_YZ.SetParallelScale(80)
            # ---------------------------------------------------------------------------------------
            bounds = self.reader.GetOutput().GetBounds()
            center0 = (bounds[1] + bounds[0]) / 2.0
            center1 = (bounds[3] + bounds[2]) / 2.0
            center2 = (bounds[5] + bounds[4]) / 2.0

            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)
            transform_YZ.RotateX(180)
            transform_YZ.RotateZ(180)
            transform_YZ.Translate(-center0, -center1, -center2)
            self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
            # ----------------------------------------------------------------------------------------
            self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                   self.id_YZ)
            self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                     self.id_YZ)
            self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
            self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
            self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
            # -------------更新冠状面---------------------------------------------------------------
            self.viewer_XZ = vtk.vtkResliceImageViewer()
            self.viewer_XZ.SetInputData(self.reader.GetOutput())
            self.viewer_XZ.SetupInteractor(self.vtkWidget3)
            self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
            self.viewer_XZ.SetSliceOrientationToXZ()
            value = self.verticalSlider_XZ.value()
            self.viewer_XZ.SetSlice(value)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            self.viewer_XZ.SliceScrollOnMouseWheelOff()
            self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
            self.camera_XZ.ParallelProjectionOn()
            self.camera_XZ.SetParallelScale(80)
            # -----------------------------------------------------------------------------------
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)
            transform_XZ.RotateZ(180)
            transform_XZ.Translate(-center0, -center1, -center2)
            self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
            # --------------------------------------------------------------------------------------
            self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                   self.id_XZ)
            self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                     self.id_XZ)
            self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
            self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)

            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()

            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()
            self.gps_enable = False
        if self.pointAction.isChecked():
            self.label_clear()
            self.pointAction.setChecked(False)
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")
        if self.labelBoxAction.isChecked():
            self.label_clear()
            self.labelBoxAction.setChecked(False)
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            try:
                for i in getLastBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the Last box actor failed!!")
            try:
                for actor in getMultipleBoundingBoxActor():
                    for i in actor:
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                clearMultipleBoundingBoxActor()
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")
            clearMultipleUndoStack()
            clearMultipleRedoStack()

        setFileIsEmpty(False)
        setIsPutImplant(False)
        setIsGenerateImplant(False)
        setAnchorPointIsComplete(False)
        setIsAdjust(False)
        setDirPath(path)
        # ------------更新横断面、矢状面和冠状面窗口数据----------------------------
        self.pathDicomDir = getDirPath()
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.pathDicomDir)
        self.reader.Update()
        self.dims = self.reader.GetOutput().GetDimensions()
        self.dicomdata, self.header = load(self.pathDicomDir)
        self.spacing = self.reader.GetOutput().GetSpacing()
        if self.spacing[2] == 0:
            newspacing = (self.spacing[0], self.spacing[1], 1)
            self.reader.GetOutput().SetSpacing(newspacing)
            self.reader.Update()
        # -------------创建分割目标矩阵--------------------------------------------
        self.segmentation_Result = np.zeros_like(self.dicomdata)
        # -------------------更新横断面------------------------------------------
        self.viewer_XY = vtk.vtkResliceImageViewer()
        self.viewer_XY.SetInputData(self.reader.GetOutput())
        self.viewer_XY.SetupInteractor(self.vtkWidget)
        self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.viewer_XY.SetSliceOrientationToXY()
        self.viewer_XY.Render()
        self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
        self.camera_XY.ParallelProjectionOn()  # 开启平行投影
        self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
        self.camera_XY_focalPoint = self.camera_XY.GetFocalPoint()
        self.camera_XY_position = self.camera_XY.GetPosition()
        self.viewer_XY.SliceScrollOnMouseWheelOff()
        self.viewer_XY.UpdateDisplayExtent()
        self.viewer_XY.Render()
        # --------------------------------------------------------------------------------------
        self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
        self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
        self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
        # slider属性
        self.value_XY = self.viewer_XY.GetSlice()
        self.maxSlice1 = self.viewer_XY.GetSliceMax()
        self.verticalSlider_XY.setMaximum(self.maxSlice1)
        self.verticalSlider_XY.setMinimum(0)
        self.verticalSlider_XY.setSingleStep(1)
        self.verticalSlider_XY.setValue(self.value_XY)
        self.verticalSlider_XY.valueChanged.connect(self.valuechange)
        self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))
        # -------------------更新矢状面------------------------------------------------------------
        self.viewer_YZ = vtk.vtkResliceImageViewer()
        self.viewer_YZ.SetInputData(self.reader.GetOutput())
        self.viewer_YZ.SetupInteractor(self.vtkWidget2)
        self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
        self.viewer_YZ.SetSliceOrientationToYZ()
        self.viewer_YZ.UpdateDisplayExtent()
        self.viewer_YZ.Render()
        self.viewer_YZ.SliceScrollOnMouseWheelOff()
        self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
        self.camera_YZ.ParallelProjectionOn()
        self.camera_YZ.SetParallelScale(80)
        self.camera_YZ_focalPoint = self.camera_YZ.GetFocalPoint()
        self.camera_YZ_position = self.camera_YZ.GetPosition()
        # ---------------------------------------------------------------------------------------
        bounds = self.reader.GetOutput().GetBounds()
        center0 = (bounds[1] + bounds[0]) / 2.0
        center1 = (bounds[3] + bounds[2]) / 2.0
        center2 = (bounds[5] + bounds[4]) / 2.0
        transform_YZ = vtk.vtkTransform()
        transform_YZ.Translate(center0, center1, center2)
        transform_YZ.RotateX(180)
        transform_YZ.RotateZ(180)
        transform_YZ.Translate(-center0, -center1, -center2)
        self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
        # ----------------------------------------------------------------------------------------
        self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
        self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
        self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
        self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
        self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
        # -----------------------------------------------------------------------------------
        self.maxSlice2 = self.viewer_YZ.GetSliceMax()
        self.verticalSlider_YZ.setMinimum(0)
        self.verticalSlider_YZ.setMaximum(self.maxSlice2)
        self.verticalSlider_YZ.setSingleStep(1)
        self.value_YZ = self.viewer_YZ.GetSlice()
        self.verticalSlider_YZ.setValue(self.value_YZ)
        self.verticalSlider_YZ.valueChanged.connect(self.valuechange2)
        self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
        # -------------更新冠状面---------------------------------------------------------------
        self.viewer_XZ = vtk.vtkResliceImageViewer()
        self.viewer_XZ.SetInputData(self.reader.GetOutput())
        self.viewer_XZ.SetupInteractor(self.vtkWidget3)
        self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
        self.viewer_XZ.SetSliceOrientationToXZ()
        self.viewer_XZ.UpdateDisplayExtent()
        self.viewer_XZ.Render()
        self.viewer_XZ.SliceScrollOnMouseWheelOff()
        self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
        self.camera_XZ.ParallelProjectionOn()
        self.camera_XZ.SetParallelScale(80)
        self.camera_XZ_focalPoint = self.camera_XZ.GetFocalPoint()
        self.camera_XZ_position = self.camera_XZ.GetPosition()
        # -----------------------------------------------------------------------------------
        transform_XZ = vtk.vtkTransform()
        transform_XZ.Translate(center0, center1, center2)
        transform_XZ.RotateY(180)
        transform_XZ.RotateZ(180)
        transform_XZ.Translate(-center0, -center1, -center2)
        self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
        # --------------------------------------------------------------------------------------
        self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
        self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
        self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
        self.viewer_XZ.Render()
        self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
        self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)
        # ---------------------------------------------------------------------------------------
        self.maxSlice3 = self.viewer_XZ.GetSliceMax()
        self.verticalSlider_XZ.setMinimum(0)
        self.verticalSlider_XZ.setMaximum(self.maxSlice3)
        self.verticalSlider_XZ.setSingleStep(1)
        self.value_XZ = self.viewer_XZ.GetSlice()
        self.verticalSlider_XZ.setValue(self.value_XZ)
        self.verticalSlider_XZ.valueChanged.connect(self.valuechange3)
        self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
        window, level = LevelAndWidth(self)
        print(window)
        print(level)
        self.window_width_slider.setValue(window)
        self.window_level_slider.setValue(level)
        self.viewer_YZ.Render()
        self.viewer_XZ.Render()
        # ------------更新体绘制窗口数据-----------------------------------------
        try:
            self.vtkWidget4, self.iren, self.renderer_volume = render_update(self, self.vtkWidget4, self.iren, path)
        except:
            print('Creat Volume error!')

    def try_alternative_conversion(self, path, save_path):
            # 这里需要实现具体的转换逻辑
            # 为了示例，我们假设转换失败
        return False, "Alternative conversion failed"

    def load_dicom(self, dicom_path):
            # 这里需要实现加载DICOM文件的逻辑
        print(f"Loading DICOM file: {dicom_path}")

    def on_actionAdd_IM0BIM_Data(self):
        self.dataformat = 'IM0'
        print("选择IM0BIM文件")
        old_path = getDirPath()
        path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "", "*.IM0;;*.BIM")
        if path == "":
            if old_path == 'F:\CBCT_Register_version_12_7\testdata\\40':
                setFileIsEmpty(True)
                return
            else:
                path = old_path
        path = path[0]
        self.IM0path = path
        print(path)
        subname = path.split('/')[-1]
        subname = subname.split('.')[0]
        print(subname)
        # ----------------------IM0转化为DICOM-----------------------------------------------
        self.save_dicompath_temp = self.outputpath + subname + '_temp/'
        if not os.path.exists(self.save_dicompath_temp):
            os.mkdir(self.save_dicompath_temp)
        else:
            for file in glob.glob(self.save_dicompath_temp + '*.dcm'):
                os.remove(file)
        os.system('mipg2dicom ' + path + ' ' + self.save_dicompath_temp)
        # -----------------处理dicom文件---------------------------------------------------
        self.save_dicompath = os.path.join(self.outputpath, subname)

        try:
            if not os.path.exists(self.save_dicompath):
                os.makedirs(self.save_dicompath)
            else:
                # 清除现有DCM文件
                for file in glob.glob(os.path.join(self.save_dicompath, '*.dcm')):
                    try:
                        os.remove(file)
                    except Exception as e:
                        print(f"删除文件{file}时出错: {e}")

            # 处理每个DICOM切片
            dicom_files.sort()
            number_slices = len(dicom_files)

            for slice_ in range(number_slices):
                try:
                    dicom_file = pydicom.dcmread(dicom_files[slice_])
                    self.SliceThickness = dicom_file.SliceThickness
                    convertNsave(dicom_file, './image_dcm/DCT0000.dcm', self.save_dicompath, slice_)
                except Exception as e:
                    print(f"处理切片{slice_}时出错: {e}")
                    continue

        except Exception as e:
            print(f"DICOM处理过程中出错: {e}")
            return
        # ---------------------------------------------------------------------------------
        self.action_ruler.setChecked(False)
        self.action_paint.setChecked(False)
        self.action_pixel.setChecked(False)
        self.action_polyline.setChecked(False)
        self.action_angle.setChecked(False)
        self.action_crosshair.setChecked(False)
        self.action_dragging_image.setChecked(False)
        self.pointAction.setChecked(False)
        self.labelBoxAction.setChecked(False)
        self.saveResultAction.setChecked(False)
        self.toolBar.update()
        # 获取目录下的所有文件名
        files = os.listdir(self.save_dicompath)
        # 检查是否存在DCM文件
        dcm_files_exist = any(file.endswith(".dcm") for file in files)
        if not dcm_files_exist:
            print("该目录下没有DCM文件数据")
            return
        # 清除之前的直线
        if self.ruler_enable == True:
            for ruler1 in self.distance_widgets_1:
                ruler1.Off()
            self.distance_widgets_1.clear()  # 清空列表
            for ruler2 in self.distance_widgets_2:
                ruler2.Off()
            self.distance_widgets_2.clear()  # 清空列表
            for ruler3 in self.distance_widgets_3:
                ruler3.Off()
            self.distance_widgets_3.clear()  # 清空列表
            self.ruler_enable = False
        # 清除之前的角度测量
        if self.angle_enable == True:
            self.angleWidget1.Off()
            self.angleWidget2.Off()
            self.angleWidget3.Off()
            self.angle_enable = False
        if self.paint_enable == True:
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            try:
                for i in getActors_paint():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_YZ.GetRenderer().RemoveActor(i)
                self.viewer_YZ.Render()
            except:
                print('Close viewer_YZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_XZ.GetRenderer().RemoveActor(i)
                self.viewer_XZ.Render()
            except:
                print('Close viewer_XZ actor_paint Failed!!!')
            self.paint_enable = False
        if self.polyline_enable == True:
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            # ---------------------------------------------------------------
            try:
                for i in getActors_paint():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_YZ.GetRenderer().RemoveActor(i)
                self.viewer_YZ.Render()
            except:
                print('Close viewer_YZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_XZ.GetRenderer().RemoveActor(i)
                self.viewer_XZ.Render()
            except:
                print('Close viewer_XZ actor_paint Failed!!!')
            self.polyline_enable = False
        if self.pixel_enable == True:
            self.vtkWidget.setToolTip('')
            self.vtkWidget2.setToolTip('')
            self.vtkWidget3.setToolTip('')
            self.imagestyle1.RemoveObservers("MouseMoveEvent")
            self.imagestyle2.RemoveObservers("MouseMoveEvent")
            self.imagestyle3.RemoveObservers("MouseMoveEvent")
            self.pixel_enable = False
        if self.gps_enable == True:
            self.action_crosshair.setChecked(False)
            # ------------------------------------------------------------------------
            self.viewer_XY.GetResliceCursorWidget().EnabledOff()
            self.viewer_YZ.GetResliceCursorWidget().EnabledOff()
            self.viewer_XZ.GetResliceCursorWidget().EnabledOff()

            self.window_level = self.window_level_slider.value()
            self.window_width = self.window_width_slider.value()
            print(self.window_level, self.window_width)
            self.pathDicomDir = getDirPath()
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.pathDicomDir)
            self.reader.Update()
            self.dims = self.reader.GetOutput().GetDimensions()
            self.dicomdata, self.header = load(self.pathDicomDir)
            # -------------------更新横断面------------------------------------------
            self.viewer_XY = vtk.vtkResliceImageViewer()
            self.viewer_XY.SetInputData(self.reader.GetOutput())
            self.viewer_XY.SetupInteractor(self.vtkWidget)
            self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
            self.viewer_XY.SetSliceOrientationToXY()
            value = self.verticalSlider_XY.value()
            self.viewer_XY.SetSlice(value)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
            self.camera_XY.ParallelProjectionOn()  # 开启平行投影
            self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
            self.viewer_XY.SliceScrollOnMouseWheelOff()
            self.viewer_XY.Render()
            # --------------------------------------------------------------------------------------
            self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                   self.id_XY)
            self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                     self.id_XY)
            self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
            # -------------------更新矢状面------------------------------------------------------------
            self.viewer_YZ = vtk.vtkResliceImageViewer()
            self.viewer_YZ.SetInputData(self.reader.GetOutput())
            self.viewer_YZ.SetupInteractor(self.vtkWidget2)
            self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
            self.viewer_YZ.SetSliceOrientationToYZ()
            value = self.verticalSlider_YZ.value()
            self.viewer_YZ.SetSlice(value)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            self.viewer_YZ.SliceScrollOnMouseWheelOff()
            self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
            self.camera_YZ.ParallelProjectionOn()
            self.camera_YZ.SetParallelScale(80)
            # ---------------------------------------------------------------------------------------
            bounds = self.reader.GetOutput().GetBounds()
            center0 = (bounds[1] + bounds[0]) / 2.0
            center1 = (bounds[3] + bounds[2]) / 2.0
            center2 = (bounds[5] + bounds[4]) / 2.0

            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)
            transform_YZ.RotateX(180)
            transform_YZ.RotateZ(180)
            transform_YZ.Translate(-center0, -center1, -center2)
            self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
            # ----------------------------------------------------------------------------------------
            self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                   self.id_YZ)
            self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                     self.id_YZ)
            self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
            self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
            self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
            # -------------更新冠状面---------------------------------------------------------------
            self.viewer_XZ = vtk.vtkResliceImageViewer()
            self.viewer_XZ.SetInputData(self.reader.GetOutput())
            self.viewer_XZ.SetupInteractor(self.vtkWidget3)
            self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
            self.viewer_XZ.SetSliceOrientationToXZ()
            value = self.verticalSlider_XZ.value()
            self.viewer_XZ.SetSlice(value)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            self.viewer_XZ.SliceScrollOnMouseWheelOff()
            self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
            self.camera_XZ.ParallelProjectionOn()
            self.camera_XZ.SetParallelScale(80)
            # -----------------------------------------------------------------------------------
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)
            transform_XZ.RotateZ(180)
            transform_XZ.Translate(-center0, -center1, -center2)
            self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
            # --------------------------------------------------------------------------------------
            self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                   self.id_XZ)
            self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                     self.id_XZ)
            self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
            self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)

            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()

            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()
            self.gps_enable = False
        if self.pointAction.isChecked():
            self.label_clear()
            self.pointAction.setChecked(False)
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")
        if self.labelBoxAction.isChecked():
            self.label_clear()
            self.labelBoxAction.setChecked(False)
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            try:
                for i in getLastBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the Last box actor failed!!")
            try:
                for actor in getMultipleBoundingBoxActor():
                    for i in actor:
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                clearMultipleBoundingBoxActor()
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")
            clearMultipleUndoStack()
            clearMultipleRedoStack()
        # ------------------------------------------------------------------
        setFileIsEmpty(False)
        setIsPutImplant(False)
        setIsGenerateImplant(False)
        setAnchorPointIsComplete(False)
        setIsAdjust(False)
        setDirPath(self.save_dicompath)
        # ------------更新横断面、矢状面和冠状面窗口数据----------------------------
        self.pathDicomDir = getDirPath()
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.pathDicomDir)
        self.reader.Update()
        self.dims = self.reader.GetOutput().GetDimensions()
        self.dicomdata, self.header = load(self.pathDicomDir)
        self.spacing = self.reader.GetOutput().GetSpacing()
        if self.spacing[2] == 0:
            print(self.SliceThickness)
            newspacing = (self.spacing[0], self.spacing[1], self.SliceThickness)
            self.reader.GetOutput().SetSpacing(newspacing)
            self.reader.Update()
        # -------------创建分割目标矩阵--------------------------------------------
        self.segmentation_Result = np.zeros_like(self.dicomdata)
        # ----------------------------------------------------------------------
        # -------------------更新横断面------------------------------------------
        self.viewer_XY = vtk.vtkResliceImageViewer()
        self.viewer_XY.SetInputData(self.reader.GetOutput())
        self.viewer_XY.SetupInteractor(self.vtkWidget)
        self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.viewer_XY.SetSliceOrientationToXY()
        self.viewer_XY.Render()
        self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
        self.camera_XY.ParallelProjectionOn()  # 开启平行投影
        self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
        self.camera_XY_focalPoint = self.camera_XY.GetFocalPoint()
        self.camera_XY_position = self.camera_XY.GetPosition()
        self.viewer_XY.SliceScrollOnMouseWheelOff()
        self.viewer_XY.UpdateDisplayExtent()
        self.viewer_XY.Render()
        # --------------------------------------------------------------------------------------
        self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
        self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
        self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
        # slider属性
        self.value_XY = self.viewer_XY.GetSlice()
        self.maxSlice1 = self.viewer_XY.GetSliceMax()
        self.verticalSlider_XY.setMaximum(self.maxSlice1)
        self.verticalSlider_XY.setMinimum(0)
        self.verticalSlider_XY.setSingleStep(1)
        self.verticalSlider_XY.setValue(self.value_XY)
        self.verticalSlider_XY.valueChanged.connect(self.valuechange)
        self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))
        # -------------------更新矢状面------------------------------------------------------------
        self.viewer_YZ = vtk.vtkResliceImageViewer()
        self.viewer_YZ.SetInputData(self.reader.GetOutput())
        self.viewer_YZ.SetupInteractor(self.vtkWidget2)
        self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
        self.viewer_YZ.SetSliceOrientationToYZ()
        self.viewer_YZ.UpdateDisplayExtent()
        self.viewer_YZ.Render()
        self.viewer_YZ.SliceScrollOnMouseWheelOff()
        self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
        self.camera_YZ.ParallelProjectionOn()
        self.camera_YZ.SetParallelScale(80)
        self.camera_YZ_focalPoint = self.camera_YZ.GetFocalPoint()
        self.camera_YZ_position = self.camera_YZ.GetPosition()
        # ---------------------------------------------------------------------------------------
        # bounds = self.reader.GetOutput().GetBounds()
        # center0 = (bounds[1] + bounds[0]) / 2.0
        # center1 = (bounds[3] + bounds[2]) / 2.0
        # center2 = (bounds[5] + bounds[4]) / 2.0
        # transform_YZ = vtk.vtkTransform()
        # transform_YZ.Translate(center0, center1, center2)
        # transform_YZ.RotateX(90)
        # transform_YZ.RotateZ(180)
        # transform_YZ.Translate(-center0, -center1, -center2)
        # self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
        # ----------------------------------------------------------------------------------------
        self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
        self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
        self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
        self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
        self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
        # -----------------------------------------------------------------------------------
        self.maxSlice2 = self.viewer_YZ.GetSliceMax()
        self.verticalSlider_YZ.setMinimum(0)
        self.verticalSlider_YZ.setMaximum(self.maxSlice2)
        self.verticalSlider_YZ.setSingleStep(1)
        self.value_YZ = self.viewer_YZ.GetSlice()
        self.verticalSlider_YZ.setValue(self.value_YZ)
        self.verticalSlider_YZ.valueChanged.connect(self.valuechange2)
        self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
        # -------------更新冠状面---------------------------------------------------------------
        self.viewer_XZ = vtk.vtkResliceImageViewer()
        self.viewer_XZ.SetInputData(self.reader.GetOutput())
        self.viewer_XZ.SetupInteractor(self.vtkWidget3)
        self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
        self.viewer_XZ.SetSliceOrientationToXZ()
        self.viewer_XZ.UpdateDisplayExtent()
        self.viewer_XZ.Render()
        self.viewer_XZ.SliceScrollOnMouseWheelOff()
        self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
        self.camera_XZ.ParallelProjectionOn()
        self.camera_XZ.SetParallelScale(80)
        self.camera_XZ_focalPoint = self.camera_XZ.GetFocalPoint()
        self.camera_XZ_position = self.camera_XZ.GetPosition()
        # -----------------------------------------------------------------------------------
        # transform_XZ = vtk.vtkTransform()
        # transform_XZ.Translate(center0, center1, center2)
        # transform_XZ.RotateY(90)
        # transform_XZ.RotateZ(180)
        # transform_XZ.Translate(-center0, -center1, -center2)
        # self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
        # --------------------------------------------------------------------------------------
        self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
        self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
        self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
        self.viewer_XZ.Render()
        self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
        self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)
        # ---------------------------------------------------------------------------------------
        self.maxSlice3 = self.viewer_XZ.GetSliceMax()
        self.verticalSlider_XZ.setMinimum(0)
        self.verticalSlider_XZ.setMaximum(self.maxSlice3)
        self.verticalSlider_XZ.setSingleStep(1)
        self.value_XZ = self.viewer_XZ.GetSlice()
        self.verticalSlider_XZ.setValue(self.value_XZ)
        self.verticalSlider_XZ.valueChanged.connect(self.valuechange3)
        self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
        window, level = LevelAndWidth(self)
        print(window)
        print(level)
        self.window_width_slider.setValue(window)
        self.window_level_slider.setValue(level)
        self.viewer_YZ.Render()
        self.viewer_XZ.Render()
        # ------------更新体绘制窗口数据-----------------------------------------
        try:
            self.vtkWidget4, self.iren, self.renderer_volume = render_update(self, self.vtkWidget4, self.iren, path)
        except Exception as e:
            print(f'体绘制错误: {e}')

    def LoadSTL(self, filename):
        # [保留原始的LoadSTL实现]
        bounds = self.reader.GetOutput().GetBounds()
        self.center0 = (bounds[1] + bounds[0]) / 2.0
        self.center1 = (bounds[3] + bounds[2]) / 2.0
        self.center2 = (bounds[5] + bounds[4]) / 2.0
        transform = vtk.vtkTransform()
        transform.Translate(self.center0, self.center1, self.center2)

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.SetUserTransform(transform)
        return actor

    def on_actionAdd_STL_Data(self):
        print("选择STL文件")
        path = QtWidgets.QFileDialog.getOpenFileName(None, "选择STL文件", filter="*.stl")
        print(path)
        actor_stl = self.LoadSTL(path[0])

        self.reader_stl_iren.SetInteractorStyle(self.reader_stl_style)
        self.reader_stl_renderer.AddActor(actor_stl)
        self.vtkWidget4.Render()
        self.reader_stl_renderer.ResetCamera()
        self.reader_stl_iren.Initialize()

    def on_action_ruler(self):
        print("直尺功能")
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用直尺功能")
            return
        if self.ruler_enable == False:
            print("测量距离")
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.pointAction.isChecked():
                self.label_clear()
                self.pointAction.setChecked(False)
                self.widget_labels.hide()
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                try:
                    self.vviewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.vviewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                self.labelBoxAction.setChecked(False)
                clearMultipleUndoStack()
                clearMultipleRedoStack()

            ruler1 = vtk.vtkDistanceWidget()
            ruler2 = vtk.vtkDistanceWidget()
            ruler3 = vtk.vtkDistanceWidget()

            iren1 = self.vtkWidget.GetRenderWindow().GetInteractor()
            iren2 = self.vtkWidget2.GetRenderWindow().GetInteractor()
            iren3 = self.vtkWidget3.GetRenderWindow().GetInteractor()

            ruler1.SetInteractor(iren1)
            ruler2.SetInteractor(iren2)
            ruler3.SetInteractor(iren3)

            ruler1.CreateDefaultRepresentation()
            ruler2.CreateDefaultRepresentation()
            ruler3.CreateDefaultRepresentation()

            ruler1.On()
            ruler2.On()
            ruler3.On()

            self.distance_widgets_1.append(ruler1)
            self.distance_widgets_2.append(ruler2)
            self.distance_widgets_3.append(ruler3)

            self.ruler_enable = True
        else:
            for ruler1 in self.distance_widgets_1:
                ruler1.Off()
            self.distance_widgets_1.clear()  # 清空列表
            for ruler2 in self.distance_widgets_2:
                ruler2.Off()
            self.distance_widgets_2.clear()  # 清空列表
            for ruler3 in self.distance_widgets_3:
                ruler3.Off()
            self.distance_widgets_3.clear()  # 清空列表
            self.ruler_enable = False

    def on_action_paint(self):
        print("画笔功能")
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用画笔功能")
            return
        if self.paint_enable == False:
            print("画笔标注")
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.pointAction.isChecked():
                self.label_clear()
                self.widget_labels.hide()
                self.pointAction.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.labelBoxAction.setChecked(False)
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                clearMultipleUndoStack()
                clearMultipleRedoStack()
            type1 = "xy"
            self.picker = vtk.vtkPointPicker()
            self.picker.PickFromListOn()
            self.left1 = LeftButtonPressEvent(self.picker, self.viewer_XY, type1)  # 画笔
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left1)
            self.do1 = MouseMoveEvent(self.picker, self.viewer_XY, type1)
            self.viewer_XY_InteractorStyle.AddObserver("MouseMoveEvent", self.do1)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            try:
                left = LeftButtonPressEvent(self.picker, self.viewer_dicom_xy, type1)  # 画笔
                self.viewer_dicom_interactor_xy.AddObserver("LeftButtonPressEvent", left)
                do = MouseMoveEvent(self.picker, self.viewer_dicom_xy, type1)
                self.viewer_dicom_interactor_xy.AddObserver("MouseMoveEvent", do)
                self.viewer_dicom_xy.UpdateDisplayExtent()
                self.viewer_dicom_xy.Render()
            except:
                print("don't exist")

            type2 = "yz"
            self.left2 = LeftButtonPressEvent(self.picker, self.viewer_YZ, type2)  # 画笔
            self.viewer_YZ_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left2)
            self.do2 = MouseMoveEvent(self.picker, self.viewer_YZ, type2)
            self.viewer_YZ_InteractorStyle.AddObserver("MouseMoveEvent", self.do2)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            try:
                left = LeftButtonPressEvent(self.picker, self.viewer_dicom_yz, type2)  # 画笔
                self.viewer_dicom_interactor_yz.AddObserver("LeftButtonPressEvent", left)
                do = MouseMoveEvent(self.picker, self.viewer_dicom_yz, type2)
                self.viewer_dicom_interactor_yz.AddObserver("MouseMoveEvent", do)
                self.viewer_dicom_yz.UpdateDisplayExtent()
                self.viewer_dicom_yz.Render()
            except:
                print("don't exist")

            type3 = "xz"
            self.left3 = LeftButtonPressEvent(self.picker, self.viewer_XZ, type3)  # 画笔
            self.viewer_XZ_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left3)
            self.do3 = MouseMoveEvent(self.picker, self.viewer_XZ, type3)
            self.viewer_XZ_InteractorStyle.AddObserver("MouseMoveEvent", self.do3)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            try:
                left = LeftButtonPressEvent(self.picker, self.viewer_dicom_xz, type3)  # 画笔
                self.viewer_dicom_interactor_xz.AddObserver("LeftButtonPressEvent", left)
                do = MouseMoveEvent(self.picker, self.viewer_dicom_xz, type3)
                self.viewer_dicom_interactor_xz.AddObserver("MouseMoveEvent", do)
                self.viewer_dicom_xz.UpdateDisplayExtent()
                self.viewer_dicom_xz.Render()
            except:
                print("don't exist")
            # ------------------------------------------------------
            self.paint_enable = True
        else:
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            try:
                self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
            except:
                print("don't exist")
            try:
                for i in getActors_paint():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_YZ.GetRenderer().RemoveActor(i)
                self.viewer_YZ.Render()
            except:
                print('Close viewer_YZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_XZ.GetRenderer().RemoveActor(i)
                self.viewer_XZ.Render()
            except:
                print('Close viewer_XZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                self.viewer_dicom_xy.Render()
            except:
                print('Close viewer_dicom_xy actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_dicom_yz.GetRenderer().RemoveActor(i)
                self.viewer_dicom_yz.Render()
            except:
                print('Close viewer_dicom_xy actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_dicom_xz.GetRenderer().RemoveActor(i)
                self.viewer_dicom_xz.Render()
            except:
                print('Close viewer_dicom_xy actor_paint Failed!!!')
            # -------------------------------------------------------------
            self.paint_enable = False

    def on_action_polyline(self):
        print("折线标注功能")
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用折线标注功能")
            return
        if self.polyline_enable == False:
            print("折线标注")
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.pointAction.isChecked():
                self.label_clear()
                self.widget_labels.hide()
                self.pointAction.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_xy.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.labelBoxAction.setChecked(False)
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the single box actor failed!!")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                clearMultipleUndoStack()
                clearMultipleRedoStack()

            type1 = "xy"
            self.picker_poly = vtk.vtkPointPicker()
            self.picker_poly.PickFromListOn()
            self.poly1 = LeftButtonPressEvent_poly(self.picker_poly, self.viewer_XY, type1)  # 画折线
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonPressEvent", self.poly1)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()

            try:
                poly = LeftButtonPressEvent_poly(self.picker_poly, self.viewer_dicom_xy, type1)  # 画折线
                self.viewer_dicom_interactor_xy.AddObserver("LeftButtonPressEvent", poly)
                self.viewer_dicom_xy.UpdateDisplayExtent()
                self.viewer_dicom_xy.Render()
            except:
                print("don't exist")

            type2 = "yz"
            self.poly2 = LeftButtonPressEvent_poly(self.picker_poly, self.viewer_YZ, type2)  # 画折线
            self.viewer_YZ_InteractorStyle.AddObserver("LeftButtonPressEvent", self.poly2)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()

            try:
                poly = LeftButtonPressEvent_poly(self.picker_poly, self.viewer_dicom_yz, type2)  # 画折线
                self.viewer_dicom_interactor_yz.AddObserver("LeftButtonPressEvent", poly)
                self.viewer_dicom_yz.UpdateDisplayExtent()
                self.viewer_dicom_yz.Render()
            except:
                print("don't exist")

            type3 = "xz"
            self.poly3 = LeftButtonPressEvent_poly(self.picker_poly, self.viewer_XZ, type3)  # 画折线
            self.viewer_XZ_InteractorStyle.AddObserver("LeftButtonPressEvent", self.poly3)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()

            try:
                poly = LeftButtonPressEvent_poly(self.picker_poly, self.viewer_dicom_xz, type3)  # 画折线
                self.viewer_dicom_interactor_xz.AddObserver("LeftButtonPressEvent", poly)
                self.viewer_dicom_xz.UpdateDisplayExtent()
                self.viewer_dicom_xz.Render()
            except:
                print("don't exist")
            # ------------------------------------------------------
            self.polyline_enable = True
        else:
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            try:
                self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
            except:
                print("don't exist")
            # ---------------------------------------------------------------
            try:
                for i in getActors_paint():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_YZ.GetRenderer().RemoveActor(i)
                self.viewer_YZ.Render()
            except:
                print('Close viewer_YZ actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_XZ.GetRenderer().RemoveActor(i)
                self.viewer_XZ.Render()
            except:
                print('Close viewer_XZ actor_paint Failed!!!')
            # -------------------------------------------------------------
            try:
                for i in getActors_paint():
                    self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                self.viewer_dicom_xy.Render()
            except:
                print('Close viewer_dicom_xy actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_dicom_yz.GetRenderer().RemoveActor(i)
                self.viewer_dicom_yz.Render()
            except:
                print('Close viewer_dicom_yz actor_paint Failed!!!')
            try:
                for i in getActors_paint():
                    self.viewer_dicom_xz.GetRenderer().RemoveActor(i)
                self.viewer_dicom_xz.Render()
            except:
                print('Close viewer_dicom_xz actor_paint Failed!!!')
            self.polyline_enable = False

    def on_action_pixel(self):
        print("骨密度按钮")
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用骨密度功能")
            return
        if self.pixel_enable == False:
            print("骨密度")
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.pointAction.isChecked():
                self.label_clear()
                self.pointAction.setChecked(False)
                self.widget_labels.hide()
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_xy.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.labelBoxAction.setChecked(False)
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_xy.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the single box actor failed!!")
                clearMultipleUndoStack()
                clearMultipleRedoStack()

            self.picker1 = vtk.vtkPointPicker()
            # ------------------------------------------------------------------------------------
            dicomdata_XY = np.rot90(self.dicomdata, 2, axes=(0, 1))
            dicomdata_XY = np.rot90(dicomdata_XY, 2, axes=(0, 2))
            self.observer1 = CallBack(self.viewer_XY, self.picker1, self.vtkWidget, dicomdata_XY)
            self.imagestyle1 = self.viewer_XY.GetInteractorStyle()
            self.imagestyle1.AddObserver("MouseMoveEvent", self.observer1)
            try:
                observer = CallBack(self.viewer_dicom_xy, self.picker1, self.vtkWidget, dicomdata_XY)
                self.viewer_dicom_interactor_xy.AddObserver("MouseMoveEvent", observer)
            except:
                print("don't exist")

            self.picker2 = vtk.vtkPointPicker()
            # -----------------------------------------------------------------------------
            # dicomdata_YZ = np.rot90(self.dicomdata, 2, axes=(0, 1))
            dicomdata_YZ = np.rot90(dicomdata_XY, 2, axes=(1, 2))
            self.observer2 = CallBack(self.viewer_YZ, self.picker2, self.vtkWidget2, dicomdata_YZ)
            self.imagestyle2 = self.viewer_YZ.GetInteractorStyle()
            self.imagestyle2.AddObserver("MouseMoveEvent", self.observer2)

            try:
                observer = CallBack(self.viewer_dicom_yz, self.picker2, self.vtkWidget2, dicomdata_YZ)
                self.viewer_dicom_interactor_yz.AddObserver("MouseMoveEvent", observer)
            except:
                print("don't exist")

            self.picker3 = vtk.vtkPointPicker()
            self.observer3 = CallBack(self.viewer_XZ, self.picker3, self.vtkWidget3, self.dicomdata)
            self.imagestyle3 = self.viewer_XZ.GetInteractorStyle()
            self.imagestyle3.AddObserver("MouseMoveEvent", self.observer3)
            try:
                observer = CallBack(self.viewer_dicom_xz, self.picker3, self.vtkWidget3, self.dicomdata)
                self.viewer_dicom_interactor_xz.AddObserver("MouseMoveEvent", observer)
            except:
                print("don't exist")
            self.pixel_enable = True
        else:
            self.vtkWidget.setToolTip('')
            self.vtkWidget2.setToolTip('')
            self.vtkWidget3.setToolTip('')
            self.imagestyle1.RemoveObservers("MouseMoveEvent")
            self.imagestyle2.RemoveObservers("MouseMoveEvent")
            self.imagestyle3.RemoveObservers("MouseMoveEvent")
            try:
                self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
            except:
                print("don't esist")
            self.pixel_enable = False

    def on_action_angle(self):
        print("角度测量功能")
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用角度测量工具")
            return
        if self.angle_enable == False:
            print("角度测量")
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.pointAction.isChecked():
                self.label_clear()
                self.pointAction.setChecked(False)
                self.widget_labels.hide()
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.labelBoxAction.setChecked(False)
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")

                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the Last box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_dicom_xy.Render()
                except:
                    print("clear the single box actor failed!!")

                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                clearMultipleUndoStack()
                clearMultipleRedoStack()

            self.angleWidget1 = vtk.vtkAngleWidget()
            self.angleWidget2 = vtk.vtkAngleWidget()
            self.angleWidget3 = vtk.vtkAngleWidget()

            iren1 = self.vtkWidget.GetRenderWindow().GetInteractor()
            iren2 = self.vtkWidget2.GetRenderWindow().GetInteractor()
            iren3 = self.vtkWidget3.GetRenderWindow().GetInteractor()

            self.angleWidget1.SetInteractor(iren1)
            self.angleWidget2.SetInteractor(iren2)
            self.angleWidget3.SetInteractor(iren3)

            self.angleWidget1.On()
            self.angleWidget2.On()
            self.angleWidget3.On()

            self.angleWidget1.CreateDefaultRepresentation()
            self.angleWidget2.CreateDefaultRepresentation()
            self.angleWidget3.CreateDefaultRepresentation()

            self.angle_enable = True
        else:
            self.angleWidget1.Off()
            self.angleWidget2.Off()
            self.angleWidget3.Off()
            self.angle_enable = False

    def on_action_reset(self):
        print("复位功能")
        if getFileIsEmpty() == True:
            print("未导入文件，不能使用复位功能")
            return
        print("开始复位")
        # 使图像恢复默认大小
        self.camera_XY.SetParallelScale(80)
        self.camera_XY.SetFocalPoint(self.camera_XY_focalPoint[0], self.camera_XY_focalPoint[1],
                                     self.camera_XY_focalPoint[2])
        self.camera_XY.SetPosition(self.camera_XY_position[0], self.camera_XY_position[1], self.camera_XY_position[2])
        self.viewer_XY.UpdateDisplayExtent()
        self.viewer_XY.Render()

        try:
            self.viewer_xy_camera.SetParallelScale(80)
            self.viewer_xy_camera.SetFocalPoint(self.viewer_camera_XY_focalPoint[0],
                                                self.viewer_camera_XY_focalPoint[1],
                                                self.viewer_camera_XY_focalPoint[2])
            self.viewer_xy_camera.SetPosition(self.viewer_camera_XY_position[0], self.viewer_camera_XY_position[1],
                                              self.viewer_camera_XY_position[2])
            self.viewer_dicom_xy.UpdateDisplayExtent()
            self.viewer_dicom_xy.Render()
        except:
            print("don't exist")

        self.camera_YZ.SetParallelScale(80)
        self.camera_YZ.SetFocalPoint(self.camera_YZ_focalPoint[0], self.camera_YZ_focalPoint[1],
                                     self.camera_YZ_focalPoint[2])
        self.camera_YZ.SetPosition(self.camera_YZ_position[0], self.camera_YZ_position[1], self.camera_YZ_position[2])
        self.viewer_YZ.UpdateDisplayExtent()
        self.viewer_YZ.Render()

        try:
            self.viewer_yz_camera.SetParallelScale(80)
            self.viewer_yz_camera.SetFocalPoint(self.viewer_camera_YZ_focalPoint[0],
                                                self.viewer_camera_YZ_focalPoint[1],
                                                self.viewer_camera_YZ_focalPoint[2])
            self.viewer_yz_camera.SetPosition(self.viewer_camera_YZ_position[0], self.viewer_camera_YZ_position[1],
                                              self.viewer_camera_YZ_position[2])
            self.viewer_dicom_yz.UpdateDisplayExtent()
            self.viewer_dicom_yz.Render()
        except:
            print("don't exist")

        self.camera_XZ.SetParallelScale(80)
        self.camera_XZ.SetFocalPoint(self.camera_XZ_focalPoint[0], self.camera_XZ_focalPoint[1],
                                     self.camera_XZ_focalPoint[2])
        self.camera_XZ.SetPosition(self.camera_XZ_position[0], self.camera_XZ_position[1], self.camera_XZ_position[2])
        self.viewer_XZ.UpdateDisplayExtent()
        self.viewer_XZ.Render()

        try:
            self.viewer_xz_camera.SetParallelScale(80)
            self.viewer_xz_camera.SetFocalPoint(self.viewer_camera_XZ_focalPoint[0],
                                                self.viewer_camera_XZ_focalPoint[1],
                                                self.viewer_camera_XZ_focalPoint[2])
            self.viewer_xz_camera.SetPosition(self.viewer_camera_XZ_position[0], self.viewer_camera_XZ_position[1],
                                              self.viewer_camera_XZ_position[2])
            self.viewer_dicom_xz.UpdateDisplayExtent()
            self.viewer_dicom_xz.Render()
        except:
            print("don't exist")

    def on_action_dragging_image(self):
        if getFileIsEmpty():
            print("未导入文件")
            return
        if self.action_dragging_image.isChecked():
            self.QMainWindow.setCursor(Qt.SizeAllCursor)
            print("图像拖动")
            # 清除之前的直线
            if self.ruler_enable == True:
                self.action_ruler.setChecked(False)
                for ruler1 in self.distance_widgets_1:
                    ruler1.Off()
                self.distance_widgets_1.clear()  # 清空列表
                for ruler2 in self.distance_widgets_2:
                    ruler2.Off()
                self.distance_widgets_2.clear()  # 清空列表
                for ruler3 in self.distance_widgets_3:
                    ruler3.Off()
                self.distance_widgets_3.clear()  # 清空列表
                self.ruler_enable = False
            # 清除之前的角度测量
            if self.angle_enable == True:
                self.action_angle.setChecked(False)
                self.angleWidget1.Off()
                self.angleWidget2.Off()
                self.angleWidget3.Off()
                self.angle_enable = False
            if self.paint_enable == True:
                self.action_paint.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                # ---------------------------------------------------------------
                try:
                    for i in getActors_paint():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_YZ.GetRenderer().RemoveActor(i)
                    self.viewer_YZ.Render()
                except:
                    print('Close viewer_YZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_XZ.GetRenderer().RemoveActor(i)
                    self.viewer_XZ.Render()
                except:
                    print('Close viewer_XZ actor_paint Failed!!!')
                self.paint_enable = False
            if self.polyline_enable == True:
                self.action_polyline.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                # ---------------------------------------------------------------
                try:
                    for i in getActors_paint():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_YZ.GetRenderer().RemoveActor(i)
                    self.viewer_YZ.Render()
                except:
                    print('Close viewer_YZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_XZ.GetRenderer().RemoveActor(i)
                    self.viewer_XZ.Render()
                except:
                    print('Close viewer_XZ actor_paint Failed!!!')
                self.polyline_enable = False
            if self.pixel_enable == True:
                self.action_pixel.setChecked(False)
                self.vtkWidget.setToolTip('')
                self.vtkWidget2.setToolTip('')
                self.vtkWidget3.setToolTip('')
                self.imagestyle1.RemoveObservers("MouseMoveEvent")
                self.imagestyle2.RemoveObservers("MouseMoveEvent")
                self.imagestyle3.RemoveObservers("MouseMoveEvent")
                self.pixel_enable = False
            if self.gps_enable == True:
                self.action_crosshair.setChecked(False)
                # ------------------------------------------------------------------------
                self.viewer_XY.GetResliceCursorWidget().EnabledOff()
                self.viewer_YZ.GetResliceCursorWidget().EnabledOff()
                self.viewer_XZ.GetResliceCursorWidget().EnabledOff()

                self.window_level = self.window_level_slider.value()
                self.window_width = self.window_width_slider.value()
                print(self.window_level, self.window_width)
                self.pathDicomDir = getDirPath()
                self.reader = vtk.vtkDICOMImageReader()
                self.reader.SetDirectoryName(self.pathDicomDir)
                self.reader.Update()
                self.dims = self.reader.GetOutput().GetDimensions()
                self.dicomdata, self.header = load(self.pathDicomDir)
                # -------------------更新横断面------------------------------------------
                self.viewer_XY = vtk.vtkResliceImageViewer()
                self.viewer_XY.SetInputData(self.reader.GetOutput())
                self.viewer_XY.SetupInteractor(self.vtkWidget)
                self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
                self.viewer_XY.SetSliceOrientationToXY()
                value = self.verticalSlider_XY.value()
                self.viewer_XY.SetSlice(value)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
                self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
                self.camera_XY.ParallelProjectionOn()  # 开启平行投影
                self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
                self.viewer_XY.SliceScrollOnMouseWheelOff()
                self.viewer_XY.Render()
                # -------------------更新矢状面------------------------------------------------------------
                self.viewer_YZ = vtk.vtkResliceImageViewer()
                self.viewer_YZ.SetInputData(self.reader.GetOutput())
                self.viewer_YZ.SetupInteractor(self.vtkWidget2)
                self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
                self.viewer_YZ.SetSliceOrientationToYZ()
                value = self.verticalSlider_YZ.value()
                self.viewer_YZ.SetSlice(value)
                self.viewer_YZ.UpdateDisplayExtent()
                self.viewer_YZ.Render()
                self.viewer_YZ.SliceScrollOnMouseWheelOff()
                self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
                self.camera_YZ.ParallelProjectionOn()
                self.camera_YZ.SetParallelScale(80)
                # ---------------------------------------------------------------------------------------
                bounds = self.reader.GetOutput().GetBounds()
                center0 = (bounds[1] + bounds[0]) / 2.0
                center1 = (bounds[3] + bounds[2]) / 2.0
                center2 = (bounds[5] + bounds[4]) / 2.0

                transform_YZ = vtk.vtkTransform()
                transform_YZ.Translate(center0, center1, center2)
                transform_YZ.RotateX(180)
                transform_YZ.RotateZ(180)
                transform_YZ.Translate(-center0, -center1, -center2)
                self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)

                # -------------更新冠状面---------------------------------------------------------------
                self.viewer_XZ = vtk.vtkResliceImageViewer()
                self.viewer_XZ.SetInputData(self.reader.GetOutput())
                self.viewer_XZ.SetupInteractor(self.vtkWidget3)
                self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
                self.viewer_XZ.SetSliceOrientationToXZ()
                value = self.verticalSlider_XZ.value()
                self.viewer_XZ.SetSlice(value)
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()
                self.viewer_XZ.SliceScrollOnMouseWheelOff()
                self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
                self.camera_XZ.ParallelProjectionOn()
                self.camera_XZ.SetParallelScale(80)
                # -----------------------------------------------------------------------------------
                transform_XZ = vtk.vtkTransform()
                transform_XZ.Translate(center0, center1, center2)
                transform_XZ.RotateY(180)
                transform_XZ.RotateZ(180)
                transform_XZ.Translate(-center0, -center1, -center2)
                self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)

                self.viewer_XY.SetColorLevel(self.window_level_slider.value())
                self.viewer_XY.SetColorWindow(self.window_width_slider.value())
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
                self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
                self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
                self.viewer_YZ.UpdateDisplayExtent()
                self.viewer_YZ.Render()
                self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
                self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()

                self.vtkWidget.GetRenderWindow().Render()
                self.vtkWidget2.GetRenderWindow().Render()
                self.vtkWidget3.GetRenderWindow().Render()
                self.gps_enable = False
            if self.pointAction.isChecked():
                self.label_clear()
                self.pointAction.setChecked(False)
                self.widget_labels.hide()
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            if self.labelBoxAction.isChecked():
                self.label_clear()
                self.labelBoxAction.setChecked(False)
                self.widget_labels.hide()
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print("clear the single box actor failed!!")
                clearMultipleUndoStack()
                clearMultipleRedoStack()
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")

            self.left_dragging_XY = LeftButtonPressEvent_Dragging(self.viewer_XY)  # 画笔
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left_dragging_XY)
            self.left_release_dragging_XY = LeftButtonReleaseEvent_Dragging(self.viewer_XY)
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonReleaseEvent", self.left_release_dragging_XY)
            self.move_dragging_XY = MouseMoveEvent_Dragging(self.viewer_XY)
            self.viewer_XY_InteractorStyle.AddObserver("MouseMoveEvent", self.move_dragging_XY)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()

            self.left_dragging_YZ = LeftButtonPressEvent_Dragging(self.viewer_YZ)
            self.viewer_YZ_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left_dragging_YZ)
            self.left_release_dragging_YZ = LeftButtonReleaseEvent_Dragging(self.viewer_YZ)
            self.viewer_YZ_InteractorStyle.AddObserver("LeftButtonReleaseEvent", self.left_release_dragging_YZ)
            self.move_dragging_YZ = MouseMoveEvent_Dragging(self.viewer_YZ)
            self.viewer_YZ_InteractorStyle.AddObserver("MouseMoveEvent", self.move_dragging_YZ)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()

            self.left_dragging_XZ = LeftButtonPressEvent_Dragging(self.viewer_XZ)
            self.viewer_XZ_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left_dragging_XZ)
            self.left_release_dragging_XZ = LeftButtonReleaseEvent_Dragging(self.viewer_XZ)
            self.viewer_XZ_InteractorStyle.AddObserver("LeftButtonReleaseEvent", self.left_release_dragging_XZ)
            self.move_dragging_XZ = MouseMoveEvent_Dragging(self.viewer_XZ)
            self.viewer_XZ_InteractorStyle.AddObserver("MouseMoveEvent", self.move_dragging_XZ)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
        else:
            self.QMainWindow.setCursor(Qt.ArrowCursor)
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")

    def on_action_point(self):
        if getFileIsEmpty():
            print("文件未导入")
            return
        if self.pointAction.isChecked():
            self.widget_labels.show()
            # 关闭其他工具的使用
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.ruler_enable == True:
                self.action_ruler.setChecked(False)
                for ruler1 in self.distance_widgets_1:
                    ruler1.Off()
                self.distance_widgets_1.clear()  # 清空列表
                for ruler2 in self.distance_widgets_2:
                    ruler2.Off()
                self.distance_widgets_2.clear()  # 清空列表
                for ruler3 in self.distance_widgets_3:
                    ruler3.Off()
                self.distance_widgets_3.clear()  # 清空列表
                self.ruler_enable = False
            # 清除之前的角度测量
            if self.angle_enable == True:
                self.action_angle.setChecked(False)
                self.angleWidget1.Off()
                self.angleWidget2.Off()
                self.angleWidget3.Off()
                self.angle_enable = False
            if self.paint_enable == True:
                self.action_paint.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                # ---------------------------------------------------------------
                try:
                    for i in getActors_paint():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_YZ.GetRenderer().RemoveActor(i)
                    self.viewer_YZ.Render()
                except:
                    print('Close viewer_YZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_XZ.GetRenderer().RemoveActor(i)
                    self.viewer_XZ.Render()
                except:
                    print('Close viewer_XZ actor_paint Failed!!!')

                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print('Close viewer_dicom_xy actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_yz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_yz.Render()
                except:
                    print('Close viewer_dicom_yz actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xz.Render()
                except:
                    print('Close viewer_dicom_xz actor_paint Failed!!!')
                self.paint_enable = False
            if self.polyline_enable == True:
                self.action_polyline.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
                # ---------------------------------------------------------------
                try:
                    for i in getActors_paint():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_YZ.GetRenderer().RemoveActor(i)
                    self.viewer_YZ.Render()
                except:
                    print('Close viewer_YZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_XZ.GetRenderer().RemoveActor(i)
                    self.viewer_XZ.Render()
                except:
                    print('Close viewer_XZ actor_paint Failed!!!')

                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print('Close viewer_dicom_xy actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_yz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_yz.Render()
                except:
                    print('Close viewer_dicom_yz actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xz.Render()
                except:
                    print('Close viewer_dicom_xz actor_paint Failed!!!')
                self.polyline_enable = False
            if self.pixel_enable == True:
                self.action_pixel.setChecked(False)
                self.vtkWidget.setToolTip('')
                self.vtkWidget2.setToolTip('')
                self.vtkWidget3.setToolTip('')
                self.imagestyle1.RemoveObservers("MouseMoveEvent")
                self.imagestyle2.RemoveObservers("MouseMoveEvent")
                self.imagestyle3.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                self.pixel_enable = False
            if self.gps_enable == True:
                self.action_crosshair.setChecked(False)
                # ------------------------------------------------------------------------
                self.viewer_XY.GetResliceCursorWidget().EnabledOff()
                self.viewer_YZ.GetResliceCursorWidget().EnabledOff()
                self.viewer_XZ.GetResliceCursorWidget().EnabledOff()

                self.window_level = self.window_level_slider.value()
                self.window_width = self.window_width_slider.value()
                print(self.window_level, self.window_width)
                self.pathDicomDir = getDirPath()
                self.reader = vtk.vtkDICOMImageReader()
                self.reader.SetDirectoryName(self.pathDicomDir)
                self.reader.Update()
                self.dims = self.reader.GetOutput().GetDimensions()
                self.dicomdata, self.header = load(self.pathDicomDir)
                # -------------------更新横断面------------------------------------------
                self.viewer_XY = vtk.vtkResliceImageViewer()
                self.viewer_XY.SetInputData(self.reader.GetOutput())
                self.viewer_XY.SetupInteractor(self.vtkWidget)
                self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
                self.viewer_XY.SetSliceOrientationToXY()
                value = self.verticalSlider_XY.value()
                self.viewer_XY.SetSlice(value)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
                self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
                self.camera_XY.ParallelProjectionOn()  # 开启平行投影
                self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
                self.viewer_XY.SliceScrollOnMouseWheelOff()
                self.viewer_XY.Render()
                # --------------------------------------------------------------------------------------
                self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                       self.id_XY)
                self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                         self.id_XY)
                self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
                self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
                self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
                # -------------------更新矢状面------------------------------------------------------------
                self.viewer_YZ = vtk.vtkResliceImageViewer()
                self.viewer_YZ.SetInputData(self.reader.GetOutput())
                self.viewer_YZ.SetupInteractor(self.vtkWidget2)
                self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
                self.viewer_YZ.SetSliceOrientationToYZ()
                value = self.verticalSlider_YZ.value()
                self.viewer_YZ.SetSlice(value)
                self.viewer_YZ.UpdateDisplayExtent()
                self.viewer_YZ.Render()
                self.viewer_YZ.SliceScrollOnMouseWheelOff()
                self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
                self.camera_YZ.ParallelProjectionOn()
                self.camera_YZ.SetParallelScale(80)
                # ---------------------------------------------------------------------------------------
                bounds = self.reader.GetOutput().GetBounds()
                center0 = (bounds[1] + bounds[0]) / 2.0
                center1 = (bounds[3] + bounds[2]) / 2.0
                center2 = (bounds[5] + bounds[4]) / 2.0

                transform_YZ = vtk.vtkTransform()
                transform_YZ.Translate(center0, center1, center2)
                transform_YZ.RotateX(180)
                transform_YZ.RotateZ(180)
                transform_YZ.Translate(-center0, -center1, -center2)
                self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
                # ----------------------------------------------------------------------------------------
                self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                       self.id_YZ)
                self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                         self.id_YZ)
                self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
                self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
                self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
                # -------------更新冠状面---------------------------------------------------------------
                self.viewer_XZ = vtk.vtkResliceImageViewer()
                self.viewer_XZ.SetInputData(self.reader.GetOutput())
                self.viewer_XZ.SetupInteractor(self.vtkWidget3)
                self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
                self.viewer_XZ.SetSliceOrientationToXZ()
                value = self.verticalSlider_XZ.value()
                self.viewer_XZ.SetSlice(value)
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()
                self.viewer_XZ.SliceScrollOnMouseWheelOff()
                self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
                self.camera_XZ.ParallelProjectionOn()
                self.camera_XZ.SetParallelScale(80)
                # -----------------------------------------------------------------------------------
                transform_XZ = vtk.vtkTransform()
                transform_XZ.Translate(center0, center1, center2)
                transform_XZ.RotateY(180)
                transform_XZ.RotateZ(180)
                transform_XZ.Translate(-center0, -center1, -center2)
                self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
                # --------------------------------------------------------------------------------------
                self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                       self.id_XZ)
                self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                         self.id_XZ)
                self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()
                self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
                self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)

                self.viewer_XY.SetColorLevel(self.window_level_slider.value())
                self.viewer_XY.SetColorWindow(self.window_width_slider.value())
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
                self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
                self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
                self.viewer_YZ.UpdateDisplayExtent()
                self.viewer_YZ.Render()
                self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
                self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()

                self.vtkWidget.GetRenderWindow().Render()
                self.vtkWidget2.GetRenderWindow().Render()
                self.vtkWidget3.GetRenderWindow().Render()
                self.gps_enable = False
            self.label_clear()
            self.labelBoxAction.setChecked(False)
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            try:
                for i in getLastBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the Last box actor failed!!")
            try:
                for actor in getMultipleBoundingBoxActor():
                    for i in actor:
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                clearMultipleBoundingBoxActor()
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")
            self.wheelforward = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                  self.id_XY)
            self.wheelbackward = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                    self.id_XY)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward)
            clearMultipleUndoStack()
            clearMultipleRedoStack()

            print("开始标注 point")
            self.picker = vtk.vtkPointPicker()
            self.picker.PickFromListOn()
            self.left_press = LeftButtonPressEvent_Point(self.picker, self.viewer_XY)
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left_press)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
        else:
            print("结束标注 point")
            self.label_clear()
            self.widget_labels.hide()
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            # self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            # self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")

    def select_point_label(self):
        # 设置一个东西存 存储分割的点标签 point_segmentation_label = 0
        if self.point_label_0.isChecked():
            setSelectPointLabel1(False)
            print("select point label 0")
        else:
            setSelectPointLabel1(True)
            print("select point label 1")

    def select_box_label(self):
        if self.box_label_single.isChecked():
            setSelectSingleBoxLabel(True)
            clearMultipleUndoStack()
            clearMultipleRedoStack()
            clearMultipleBoundingBoxDict()
            # setMultipleBoundingBoxDict({})
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getLastBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for actor in getMultipleBoundingBoxActor():
                    for i in actor:
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                clearMultipleBoundingBoxActor()
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")

            print("select box label single")
        else:
            setSelectSingleBoxLabel(False)
            print("select box label multiple")
            clearSingleUndoStack()
            clearPointsRedoStack()
            setSingleBoundingBoxDict({})
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")

    def select_slice_range(self):
        if self.segmentation_type_sliceRange.isChecked():
            self.segmentation_type_none.setChecked(False)
            self.segmentation_type_sliceRange.setChecked(True)
        else:
            self.segmentation_type_sliceRange.setChecked(False)
            self.segmentation_type_none.setChecked(True)

    def on_action_labelBox(self):
        if getFileIsEmpty():
            print("文件未导入")
            return
        if self.labelBoxAction.isChecked():
            print("开始标注 labelBox")
            self.widget_labels.show()
            # 关闭其他工具的使用
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                self.QMainWindow.setCursor(Qt.ArrowCursor)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonReleaseEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonReleaseEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonReleaseEvent")
                except:
                    print("don't exist")
            if self.ruler_enable == True:
                self.action_ruler.setChecked(False)
                for ruler1 in self.distance_widgets_1:
                    ruler1.Off()
                self.distance_widgets_1.clear()  # 清空列表
                for ruler2 in self.distance_widgets_2:
                    ruler2.Off()
                self.distance_widgets_2.clear()  # 清空列表
                for ruler3 in self.distance_widgets_3:
                    ruler3.Off()
                self.distance_widgets_3.clear()  # 清空列表
                self.ruler_enable = False
            # 清除之前的角度测量
            if self.angle_enable == True:
                self.action_angle.setChecked(False)
                self.angleWidget1.Off()
                self.angleWidget2.Off()
                self.angleWidget3.Off()
                self.angle_enable = False
            if self.paint_enable == True:
                self.action_paint.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                try:
                    for i in getActors_paint():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_YZ.GetRenderer().RemoveActor(i)
                    self.viewer_YZ.Render()
                except:
                    print('Close viewer_YZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_XZ.GetRenderer().RemoveActor(i)
                    self.viewer_XZ.Render()
                except:
                    print('Close viewer_XZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print('Close viewer_dicom_xy actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_yz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_yz.Render()
                except:
                    print('Close viewer_dicom_yz actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xz.Render()
                except:
                    print('Close viewer_dicom_xz actor_paint Failed!!!')
                self.paint_enable = False
            if self.polyline_enable == True:
                self.action_polyline.setChecked(False)
                self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_YZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                self.viewer_XZ_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("LeftButtonPressEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("LeftButtonPressEvent")
                except:
                    print("don't exist")
                # ---------------------------------------------------------------
                try:
                    for i in getActors_paint():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    self.viewer_XY.Render()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_YZ.GetRenderer().RemoveActor(i)
                    self.viewer_YZ.Render()
                except:
                    print('Close viewer_YZ actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_XZ.GetRenderer().RemoveActor(i)
                    self.viewer_XZ.Render()
                except:
                    print('Close viewer_XZ actor_paint Failed!!!')

                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xy.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xy.Render()
                except:
                    print('Close viewer_dicom_xy actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_yz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_yz.Render()
                except:
                    print('Close viewer_dicom_yz actor_paint Failed!!!')
                try:
                    for i in getActors_paint():
                        self.viewer_dicom_xz.GetRenderer().RemoveActor(i)
                    self.viewer_dicom_xz.Render()
                except:
                    print('Close viewer_dicom_xz actor_paint Failed!!!')
                self.polyline_enable = False
            if self.pixel_enable == True:
                self.action_pixel.setChecked(False)
                self.vtkWidget.setToolTip('')
                self.vtkWidget2.setToolTip('')
                self.vtkWidget3.setToolTip('')
                self.imagestyle1.RemoveObservers("MouseMoveEvent")
                self.imagestyle2.RemoveObservers("MouseMoveEvent")
                self.imagestyle3.RemoveObservers("MouseMoveEvent")
                try:
                    self.viewer_dicom_interactor_xy.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_yz.RemoveObservers("MouseMoveEvent")
                    self.viewer_dicom_interactor_xz.RemoveObservers("MouseMoveEvent")
                except:
                    print("don't exist")
                self.pixel_enable = False
            if self.gps_enable == True:
                self.action_crosshair.setChecked(False)
                # ------------------------------------------------------------------------
                self.viewer_XY.GetResliceCursorWidget().EnabledOff()
                self.viewer_YZ.GetResliceCursorWidget().EnabledOff()
                self.viewer_XZ.GetResliceCursorWidget().EnabledOff()

                self.window_level = self.window_level_slider.value()
                self.window_width = self.window_width_slider.value()
                print(self.window_level, self.window_width)
                self.pathDicomDir = getDirPath()
                self.reader = vtk.vtkDICOMImageReader()
                self.reader.SetDirectoryName(self.pathDicomDir)
                self.reader.Update()
                self.dims = self.reader.GetOutput().GetDimensions()
                self.dicomdata, self.header = load(self.pathDicomDir)
                # -------------------更新横断面------------------------------------------
                self.viewer_XY = vtk.vtkResliceImageViewer()
                self.viewer_XY.SetInputData(self.reader.GetOutput())
                self.viewer_XY.SetupInteractor(self.vtkWidget)
                self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
                self.viewer_XY.SetSliceOrientationToXY()
                value = self.verticalSlider_XY.value()
                self.viewer_XY.SetSlice(value)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
                self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
                self.camera_XY.ParallelProjectionOn()  # 开启平行投影
                self.camera_XY.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
                self.viewer_XY.SliceScrollOnMouseWheelOff()
                self.viewer_XY.Render()
                # --------------------------------------------------------------------------------------
                self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                       self.id_XY)
                self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                         self.id_XY)
                self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
                self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
                self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
                # -------------------更新矢状面------------------------------------------------------------
                self.viewer_YZ = vtk.vtkResliceImageViewer()
                self.viewer_YZ.SetInputData(self.reader.GetOutput())
                self.viewer_YZ.SetupInteractor(self.vtkWidget2)
                self.viewer_YZ.SetRenderWindow(self.vtkWidget2.GetRenderWindow())
                self.viewer_YZ.SetSliceOrientationToYZ()
                value = self.verticalSlider_YZ.value()
                self.viewer_YZ.SetSlice(value)
                self.viewer_YZ.UpdateDisplayExtent()
                self.viewer_YZ.Render()
                self.viewer_YZ.SliceScrollOnMouseWheelOff()
                self.camera_YZ = self.viewer_YZ.GetRenderer().GetActiveCamera()
                self.camera_YZ.ParallelProjectionOn()
                self.camera_YZ.SetParallelScale(80)
                # ---------------------------------------------------------------------------------------
                bounds = self.reader.GetOutput().GetBounds()
                center0 = (bounds[1] + bounds[0]) / 2.0
                center1 = (bounds[3] + bounds[2]) / 2.0
                center2 = (bounds[5] + bounds[4]) / 2.0

                transform_YZ = vtk.vtkTransform()
                transform_YZ.Translate(center0, center1, center2)
                transform_YZ.RotateX(180)
                transform_YZ.RotateZ(180)
                transform_YZ.Translate(-center0, -center1, -center2)
                self.viewer_YZ.GetImageActor().SetUserTransform(transform_YZ)
                # ----------------------------------------------------------------------------------------
                self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                       self.id_YZ)
                self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ,
                                                         self.id_YZ)
                self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
                self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
                self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
                # -------------更新冠状面---------------------------------------------------------------
                self.viewer_XZ = vtk.vtkResliceImageViewer()
                self.viewer_XZ.SetInputData(self.reader.GetOutput())
                self.viewer_XZ.SetupInteractor(self.vtkWidget3)
                self.viewer_XZ.SetRenderWindow(self.vtkWidget3.GetRenderWindow())
                self.viewer_XZ.SetSliceOrientationToXZ()
                value = self.verticalSlider_XZ.value()
                self.viewer_XZ.SetSlice(value)
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()
                self.viewer_XZ.SliceScrollOnMouseWheelOff()
                self.camera_XZ = self.viewer_XZ.GetRenderer().GetActiveCamera()
                self.camera_XZ.ParallelProjectionOn()
                self.camera_XZ.SetParallelScale(80)
                # -----------------------------------------------------------------------------------
                transform_XZ = vtk.vtkTransform()
                transform_XZ.Translate(center0, center1, center2)
                transform_XZ.RotateY(180)
                transform_XZ.RotateZ(180)
                transform_XZ.Translate(-center0, -center1, -center2)
                self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
                # --------------------------------------------------------------------------------------
                self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                       self.id_XZ)
                self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ,
                                                         self.id_XZ)
                self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()
                self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
                self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)

                self.viewer_XY.SetColorLevel(self.window_level_slider.value())
                self.viewer_XY.SetColorWindow(self.window_width_slider.value())
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
                self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
                self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
                self.viewer_YZ.UpdateDisplayExtent()
                self.viewer_YZ.Render()
                self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
                self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
                self.viewer_XZ.UpdateDisplayExtent()
                self.viewer_XZ.Render()

                self.vtkWidget.GetRenderWindow().Render()
                self.vtkWidget2.GetRenderWindow().Render()
                self.vtkWidget3.GetRenderWindow().Render()
                self.gps_enable = False
            self.pointAction.setChecked(False)
            try:
                for i in getPointsActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print('Close viewer_XY point actor Failed!!!')
            clearPointsActor()
            clearPointsUndoStack()
            clearPointsRedoStack()
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelForwardEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseWheelBackwardEvent")

            self.wheelforward = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                  self.id_XY)
            self.wheelbackward = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY,
                                                    self.id_XY)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward)
            self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward)

            self.picker = vtk.vtkPointPicker()
            self.picker.PickFromListOn()
            self.left_press_labelBox = LeftButtonPressEvent_labelBox(self.picker, self.viewer_XY)
            self.mouse_move_labelBox = MouseMoveEvent_labelBox(self.picker, self.viewer_XY)
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left_press_labelBox)
            self.viewer_XY_InteractorStyle.AddObserver("MouseMoveEvent", self.mouse_move_labelBox)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()

        else:
            print("结束标注 labelBox")
            self.widget_labels.hide()
            self.label_clear()
            try:
                for i in getLastBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            try:
                for actor in getMultipleBoundingBoxActor():
                    for i in actor:
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                clearMultipleBoundingBoxActor()
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            self.viewer_XY_InteractorStyle.RemoveObservers("MouseMoveEvent")
            clearMultipleUndoStack()
            clearMultipleRedoStack()

    def on_action_startSegmentation(self):
        if getFileIsEmpty():
            print("文件未导入")
            return
        if self.Flag_Load_model == False:
            print("Model has not been loaded!!")
            self.message_dialog('Model Load', "Model has not been loaded!!")
            return
        # ----------------------------------------------
        Meanv = np.mean(self.dicomdata)
        Std = np.std(self.dicomdata)
        Minv = np.min(self.dicomdata)
        Maxv = np.max(self.dicomdata)
        normal_val_min = np.max([Minv, Meanv - 2 * Std])
        normal_val_max = np.min([Maxv, Meanv + 2 * Std])
        if self.modeltype == 'Universal':
            dicomdata = np.uint8(MaxMin_normalization_Intensity(self.dicomdata, Minv, Maxv) * 255)
        else:
            dicomdata = np.uint8(MaxMin_normalization_Intensity(self.dicomdata, normal_val_min, normal_val_max) * 255)
        # dicomdata = self.dicomdata
        if self.pointAction.isChecked():
            points_dict = getPointsDict()
            index_list = [str(point[2]) for point in getPointsUndoStack()]
            for index in list(points_dict.keys()):
                if index not in index_list:
                    del points_dict[index]
            print(points_dict)
            if self.segmentation_type_sliceRange.isChecked():
                keys = sorted(int(k) for k in points_dict.keys())
                print("keys:", keys)
                for i in range(keys[0] + 1, keys[-1]):
                    if str(i) not in points_dict:
                        points_dict[str(i)] = {'points': points_dict[str(i - 1)]["points"],
                                               'label': points_dict[str(i - 1)]['label'],
                                               'image_name': f'_image_{i}.png'}
            print(points_dict)
            for index, data in points_dict.items():
                if self.dataformat == 'DICOM':
                    index_z = self.dicomdata.shape[2] - int(index) - 1
                else:
                    index_z = int(index)
                select_layer_image = dicomdata[:, :, index_z]
                select_layer_image_transpose = np.transpose(select_layer_image)
                input_points = []
                input_labels = data["label"]
                for point in data["points"]:
                    index_y = self.dicomdata.shape[1] - point[1] - 1
                    input_points.append([point[0], index_y])
                temp = point_segmentation(self.model, input_points, input_labels, select_layer_image_transpose,
                                          self.dataformat)
                self.segmentation_Result[:, :, index_z] = np.transpose(temp)
        if self.labelBoxAction.isChecked():
            if getSelectSingleBoxLabel():
                index_list = [str(point[4]) for point in getSingleUndoStack()]
                bounding_box_dict = getSingleBoundingBoxDict()
                for index in list(bounding_box_dict.keys()):
                    if index not in index_list:
                        del bounding_box_dict[index]
                print(bounding_box_dict)
                if self.segmentation_type_sliceRange.isChecked():
                    keys = sorted(int(k) for k in bounding_box_dict.keys())
                    print("keys:", keys)
                    for i in range(keys[0] + 1, keys[-1]):
                        if str(i) not in bounding_box_dict:
                            bounding_box_dict[str(i)] = {'bounding_box': bounding_box_dict[str(i - 1)]["bounding_box"],
                                                         'image_name': f'_image_{i}.png'}
                print(bounding_box_dict)
                for index, data in bounding_box_dict.items():
                    if self.dataformat == 'DICOM':
                        index_z = self.dicomdata.shape[2] - int(index) - 1
                    else:
                        index_z = int(index)
                    select_layer_image = dicomdata[:, :, index_z]
                    select_layer_image_transpose = np.transpose(select_layer_image)
                    start_x = data["bounding_box"][0]
                    start_y = self.dicomdata.shape[1] - data["bounding_box"][1] - 1
                    end_x = data["bounding_box"][2]
                    end_y = self.dicomdata.shape[1] - data["bounding_box"][3] - 1
                    input_box = [start_x, start_y, end_x, end_y]
                    temp = single_box_segmentation(self.model, input_box, select_layer_image_transpose)
                    self.segmentation_Result[:, :, index_z] = np.transpose(temp)
            else:
                index_list = [str(point[4]) for point in getMultipleUndoStack()]
                bounding_box_dict = getMultipleBoundingBoxDict()
                for index in list(bounding_box_dict.keys()):
                    if index not in index_list:
                        del bounding_box_dict[index]
                if self.segmentation_type_sliceRange.isChecked():
                    keys = sorted(int(k) for k in bounding_box_dict.keys())
                    print("keys:", keys)
                    for i in range(keys[0] + 1, keys[-1]):
                        if str(i) not in bounding_box_dict:
                            bounding_box_dict[str(i)] = {'bounding_box': bounding_box_dict[str(i - 1)]["bounding_box"],
                                                         'image_name': f'_image_{i}.png'}
                print(bounding_box_dict)
                for index, data in bounding_box_dict.items():
                    if self.dataformat == 'DICOM':
                        index_z = self.dicomdata.shape[2] - int(index) - 1
                    else:
                        index_z = int(index)
                    select_layer_image = dicomdata[:, :, index_z]
                    select_layer_image_transpose = np.transpose(select_layer_image)
                    input_box = []
                    for box in data["bounding_box"]:
                        start_x = box[0]
                        start_y = self.dicomdata.shape[1] - box[1] - 1
                        end_x = box[2]
                        end_y = self.dicomdata.shape[1] - box[3] - 1
                        input_point = [start_x, start_y, end_x, end_y]
                        input_box.append(input_point)
                    temp = multiple_box_segmentation(self.model, input_box, select_layer_image_transpose, self.args)
                    self.segmentation_Result[:, :, index_z] = np.transpose(temp)

                # clearMultipleUndoStack()
                # clearMultipleRedoStack()
                # try:
                #     for actor in getMultipleBoundingBoxActor():
                #         for i in actor:
                #             self.viewer_XY.GetRenderer().RemoveActor(i)
                #     clearMultipleBoundingBoxActor()
                #     self.viewer_XY.Render()
                # except:
                #     print("clear the single box actor failed!!")
                # try:
                #     for i in getLastBoundingBoxActor():
                #         self.viewer_XY.GetRenderer().RemoveActor(i)
                #     self.viewer_XY.Render()
                # except:
                #     print("clear the Last box actor failed!!")
        save(self.segmentation_Result, './output/' + self.subject_name + '.nii.gz', hdr=self.header)
        self.imageblend_seg_mask('./output/' + self.subject_name + '.nii.gz', self.dicomdata.shape[2] - index_z - 1)

    def on_action_saveResult(self):
        Segmentation_Result = self.segmentation_Result
        if self.dataformat == 'IM0':
            Segmentation_Result = np.transpose(Segmentation_Result, (1, 0, 2))
            Save_BIM(np.int32(Segmentation_Result), self.outputpath + self.subject_name + '_seg.BIM',
                     input_file=self.IM0path)
        else:
            save(np.int32(Segmentation_Result), self.outputpath + self.subject_name + '_seg.nii.gz', hdr=self.header)
        # -------------------save stl--------------------------------
        savestl(Segmentation_Result, self.spacing, self.subject_name, self.outputpath)

    def label_clear(self):
        try:
            for i in getPointsActor():
                self.viewer_XY.GetRenderer().RemoveActor(i)
            self.viewer_XY.Render()
        except:
            print('Close viewer_XY point actor Failed!!!')
        clearPointsActor()
        clearPointsUndoStack()
        clearPointsRedoStack()
        clearPointsDict()
        # setPointsDict({})

        if getSelectSingleBoxLabel():
            clearSingleUndoStack()
            clearPointsRedoStack()
            setSingleBoundingBoxDict({})
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")
        else:
            clearMultipleUndoStack()
            clearMultipleRedoStack()
            clearMultipleBoundingBoxDict()
            # setMultipleBoundingBoxDict({})
            try:
                for i in getSingleBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for i in getLastBoundingBoxActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            try:
                for actor in getMultipleBoundingBoxActor():
                    for i in actor:
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                clearMultipleBoundingBoxActor()
                self.viewer_XY.Render()
            except:
                print("clear the single box actor failed!!")

    def label_redo(self):
        if self.pointAction.isChecked():
            point_dict = getPointsDict()
            if len(getPointsRedoStack()) > 0:
                redo_point = getPointsRedoStack().pop()
                setPointsUndoStack(redo_point)
                if str(redo_point[2]) in point_dict:
                    point_dict[str(redo_point[2])]["points"].append([redo_point[0], redo_point[1]])
                    point_dict[str(redo_point[2])]["label"].append(redo_point[3])
                else:
                    point_dict[str(redo_point[2])] = {"points": [[redo_point[0], redo_point[1]]],
                                                      "label": [redo_point[3]],
                                                      "image_name": "_image_" + str(redo_point[2]) + ".png"}
            for point in getPointsUndoStack():
                if point[2] == self.verticalSlider_XY.value():
                    self.point_paints(point)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
        else:
            if getSelectSingleBoxLabel():
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                if getSingleRedoStack() != []:
                    setSingleUndoStack(getSingleRedoStack().pop())
                for data in getSingleUndoStack():
                    if data[4] == self.verticalSlider_XY.value():
                        self.actor_list = []
                        self.drwa_single_bounding_box(data)
                        setSingleBoundingBoxActor(self.actor_list)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()

            else:
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                boundingbox_dict = getMultipleBoundingBoxDict()
                print(boundingbox_dict)
                print(getMultipleRedoStack())
                if len(getMultipleRedoStack()) > 0:
                    redo_box = getMultipleRedoStack().pop()
                    print("redo_box:", redo_box)
                    setMultipleUndoStack(redo_box)
                    if str(redo_box[4]) in boundingbox_dict:
                        boundingbox_dict[str(redo_box[4])]["bounding_box"].append(redo_box)
                    else:
                        boundingbox_dict[str(redo_box[4])] = {"bounding_box": [redo_box],
                                                              "image_name": "_image_" + str(redo_box[4]) + ".png"}
                print(getMultipleUndoStack())
                print(getMultipleBoundingBoxDict())
                for data in getMultipleUndoStack():
                    if data[4] == self.verticalSlider_XY.value():
                        self.actor_list = []
                        self.drwa_single_bounding_box(data)
                        setMultipleBoundingBoxActor(self.actor_list)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()

    def label_undo(self):
        if self.pointAction.isChecked():
            point_dict = getPointsDict()
            try:
                for i in getPointsActor():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            if len(getPointsUndoStack()) > 0:
                undo_point = getPointsUndoStack().pop()
                setPointsRedoStack(undo_point)
                point_to_remove = [undo_point[0], undo_point[1]]
                for i, point in enumerate(point_dict[str(undo_point[2])]['points']):
                    # 检查点的坐标是否与要删除的点匹配
                    if point == point_to_remove:
                        # 删除该点
                        del point_dict[str(undo_point[2])]['points'][i]
                        # 删除标签
                        del point_dict[str(undo_point[2])]['label'][i]

            for point in getPointsUndoStack():
                if point[2] == self.verticalSlider_XY.value():
                    self.point_paints(point)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
        else:
            if getSelectSingleBoxLabel():
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                single_boundingBox_actor = getSingleUndoStack()
                if getSingleUndoStack() != []:
                    setSingleRedoStack(getSingleUndoStack().pop())
                for data in getSingleUndoStack():
                    if data[4] == self.verticalSlider_XY.value():
                        self.actor_list = []
                        self.drwa_single_bounding_box(data)
                        setSingleBoundingBoxActor(self.actor_list)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()
            else:
                try:
                    for i in getSingleBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for i in getLastBoundingBoxActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                except:
                    print('Close viewer_XY actor_paint Failed!!!')
                try:
                    for actor in getMultipleBoundingBoxActor():
                        for i in actor:
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    clearMultipleBoundingBoxActor()
                except:
                    print('Close viewer_XY actor_paint Failed!!!')

                bounding_box = getMultipleBoundingBoxDict()
                if getMultipleUndoStack() != []:
                    undo_point = getMultipleUndoStack().pop()
                    setMultipleRedoStack(undo_point)
                    for i, point in enumerate(bounding_box[str(undo_point[4])]['bounding_box']):
                        # 检查点的坐标是否与要删除的点匹配
                        if point == undo_point:
                            # 删除该点
                            del bounding_box[str(undo_point[4])]['bounding_box'][i]

                for data in getMultipleUndoStack():
                    if data[4] == self.verticalSlider_XY.value():
                        self.actor_list = []
                        self.drwa_single_bounding_box(data)
                        setMultipleBoundingBoxActor(self.actor_list)
                self.viewer_XY.UpdateDisplayExtent()
                self.viewer_XY.Render()

    def point_paints(self, point):
        origin = self.viewer_XY.GetInput().GetOrigin()
        spacing = self.viewer_XY.GetInput().GetSpacing()
        print("point_spacing,", spacing)

        point_x = point[0] * spacing[0] + origin[0]
        point_y = point[1] * spacing[1] + origin[1]
        point_z = point[2] * spacing[2] + origin[2]

        square = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.InsertNextPoint(point_x - 1, point_y + 1, point_z + 1)
        points.InsertNextPoint(point_x + 1, point_y + 1, point_z + 1)
        points.InsertNextPoint(point_x + 1, point_y - 1, point_z + 1)
        points.InsertNextPoint(point_x - 1, point_y - 1, point_z + 1)

        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        polygon.GetPointIds().SetId(0, 0)
        polygon.GetPointIds().SetId(1, 1)
        polygon.GetPointIds().SetId(2, 2)
        polygon.GetPointIds().SetId(3, 3)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)

        square.SetPoints(points)
        square.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(square)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 1, 0)
        setPointsActor(actor)
        self.viewer_XY.GetRenderer().AddActor(actor)

    def SetLine(self, point1, point2):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(point1)
        lineSource.SetPoint2(point2)
        lineSource.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(lineSource.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2)
        actor.GetProperty().SetColor(0.0, 1.0, 1.0)
        self.actor_list.append(actor)
        self.viewer_XY.GetRenderer().AddActor(actor)

    def drwa_single_bounding_box(self, data):
        origin = self.viewer_XY.GetInput().GetOrigin()
        spacing = self.viewer_XY.GetInput().GetSpacing()

        start_pointX = data[0] * spacing[0] + origin[0]
        start_pointY = data[1] * spacing[1] + origin[1]
        end_pointX = data[2] * spacing[0] + origin[0]
        end_pointY = data[3] * spacing[1] + origin[1]
        point_Z = data[4] * spacing[2] + origin[2] + 1

        start = [start_pointX, start_pointY]
        end = [end_pointX, end_pointY]

        left = [0, 0]
        right = [0, 0]

        left[0] = start[0] if start[0] <= end[0] else end[0]
        left[1] = start[1] if start[1] <= end[1] else end[1]

        right[0] = start[0] if start[0] > end[0] else end[0]
        right[1] = start[1] if start[1] > end[1] else end[1]

        point1 = [left[0], left[1], point_Z]
        point2 = [left[0], right[1], point_Z]
        point3 = [right[0], right[1], point_Z]
        point4 = [right[0], left[1], point_Z]

        self.SetLine(point1, point2)
        self.SetLine(point2, point3)
        self.SetLine(point3, point4)
        self.SetLine(point4, point1)


if __name__ == "__main__":
    import sys

    print("文件是否为空：", getFileIsEmpty())
    setDirPath("F:\CBCT_Register_version_12_7\testdata\\40")

    app = QtWidgets.QApplication(sys.argv)
    Widget = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Widget)
    Widget.show()
    sys.exit(app.exec_())
