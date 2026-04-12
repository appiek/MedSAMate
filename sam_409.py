# -*- coding: utf-8 -*-
# ---------------------------------------------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 重复初始化问题
# ---------------------------------------------------------
from PyQt5.QtCore import QPoint, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from types import SimpleNamespace as Namespace
# ---------------------------------------------------------
import glob
import sys
from argparse import Namespace
import matplotlib.pyplot as plt
import pyautogui
import numpy as np
import nibabel as nib
from globalVariables import *  # 自定义全局变量
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from volume import volume, render_update  # 体绘制相关函数
import vtk.util.numpy_support as numpy_support
from callback import CallBack  # 回调函数
from interactor import *  # 交互器类
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
from skimage.measure import marching_cubes  # 提取3D表面
from scipy import ndimage  # 图像处理
# --------------------------------------------------------------
import torch
import torch.nn as nn
# --------------------------------------------------------------
from sam_med2d_funcs import *  # MedSAMate相关工具函数
from toolbar_data_system import (  # 工具栏数据管理系统
    ToolbarDataManager, InfoPanel, BaseTool, 
    DataType, ToolDataModel, RulerTool, AngleTool
)
from custom_model_loader import (  # 自定义模型加载器
    CustomModelLoader, 
    CustomModelInfo, 
    ModelInfoDialog,
    RecentModelsMenu,
    ModelConfigParser,
    create_default_config_template
)
from pathlib import Path  # 用于路径处理
from history_file_manager import (  # 历史文件记录管理器
    HistoryFileManager,
    HistoryFileListWidget
)

error = vtk.vtkOutputWindow()
error.SetGlobalWarningDisplay(0)  # 关闭vtk报错信息


def BoundingBox_Coordinate(Seg, offset=20):
    """计算分割结果的3D边界框坐标"""
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
    mask_sum = np.sum(Seg > 0, axis=(0, 1))
    mask_id_list = np.where(mask_sum > 0)
    mask_id_low = min(np.max(mask_id_list) + offset, C)
    mask_id_up = max(np.min(mask_id_list) - offset, 0)

    return TL_x, BR_x, TL_y, BR_y, mask_id_up, mask_id_low


def savestl(mask, spacing, subjname, savepath, smoothfactor=2.0):
    """将分割结果转换为STL格式3D模型"""
    print(f"[STL] Starting STL generation for {subjname}...")
    
    # 检查分割结果是否有效
    if mask is None or np.sum(mask) == 0:
        raise ValueError("Empty mask, cannot generate STL")
    
    print(f"[STL] Mask shape: {mask.shape}, spacing: {spacing}")
    
    TL_x, BR_x, TL_y, BR_y, mask_id_up, mask_id_low = BoundingBox_Coordinate(mask)
    print(f"[STL] Bounding box: X[{TL_x}:{BR_x}], Y[{TL_y}:{BR_y}], Z[{mask_id_up}:{mask_id_low}]")
    
    # 获取边界框，裁剪掩码以加速处理
    seg = np.float32(mask[TL_x:BR_x, TL_y: BR_y, mask_id_up:mask_id_low] > 0)
    print(f"[STL] Cropped seg shape: {seg.shape}, min={seg.min()}, max={seg.max()}")
    
    # 高斯平滑
    if smoothfactor > 0:
        seg = ndimage.gaussian_filter(seg, sigma=smoothfactor, truncate=8.0)
        print(f"[STL] After smoothing: min={seg.min()}, max={seg.max()}")
    
    # 确保数据范围在 [0, 1] 内（针对 marching_cubes 的 level=0.5）
    seg_min, seg_max = seg.min(), seg.max()
    if seg_max > seg_min:
        seg = (seg - seg_min) / (seg_max - seg_min)
    else:
        seg = np.zeros_like(seg)
    
    print(f"[STL] Normalized: min={seg.min()}, max={seg.max()}")
    
    # 创建完整尺寸数组用于 marching_cubes
    Seg = np.zeros(mask.shape, dtype=np.float32)
    Seg[TL_x:BR_x, TL_y: BR_y, mask_id_up:mask_id_low] = seg
    
    # 用marching cubes提取表面
    level = 0.5
    print(f"[STL] Running marching_cubes with level={level}...")
    
    try:
        # skimage 0.20+ 返回 verts, faces, normals, values
        # 旧版本可能只返回 verts, faces, normals
        result = marching_cubes(Seg, level=level, spacing=spacing, step_size=1)
        if len(result) == 4:
            verts, faces, normals, values = result
        else:
            verts, faces, normals = result
            values = None
        print(f"[STL] marching_cubes success: {len(verts)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"[STL] marching_cubes failed: {e}")
        raise
    
    # 保存为STL
    print(f"[STL] Creating mesh and exporting...")
    surf_mesh = trimesh.Trimesh(vertices=verts, faces=faces, validate=False)
    output_path = savepath + subjname + '.stl'
    surf_mesh.export(output_path)
    print(f"[STL] Saved to: {output_path}")


def convertNsave(dicom_file_ori, file_dicom, file_dir, index=0):
    """DICOM文件格式转换和保存"""
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    """
    dicom_file = pydicom.dcmread(file_dicom)  # 读取原始QICOM
    arr = dicom_file_ori.pixel_array  # 获取像素数据
    arr = arr.astype('uint16')  # 转化为16位整数
    # 更新DICOM元数据
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"  # 单色模式
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1  # 有符号整数
    dicom_file.SliceThickness = dicom_file_ori.SliceThickness
    dicom_file.PixelSpacing = dicom_file_ori.PixelSpacing
    dicom_file.PixelData = arr.tobytes()  # 更新像素数据
    dicom_file.InstanceNumber = str(index + 1)
    # 保存
    dicom_file.save_as(os.path.join(file_dir, f'{"{:04d}".format(index + 1)}.dcm'))


def Load_IM0_BIM(inputfile):
    """加载IM0/BIM格式文件"""
    # ----------------------------------------------------------------------------------
    rootpath = '.'
    # 调用系统命令获取切片数
    # ============IM0 data transform into matlab===========
    slicenumber = os.popen('get_slicenumber ' + inputfile).read()
    slicenumber = slicenumber.split(' ')[1]
    # 转换为mat文件并读取
    os.system('exportMath ' + inputfile + ' matlab ' + rootpath + '\\temp.mat' + ' 0 ' + slicenumber)
    input_mridata = loadmat(rootpath + '\\temp.mat')['scene']
    os.remove(rootpath + '\\temp.mat')  # 清理临时文件
    return input_mridata


def Save_BIM(Img, output_file, input_file=None):
    """保存为BIM格式"""
    # ----------------------------------------------------------------------------------
    rootpath = '.'
    img_shape = np.shape(Img)
    # 保存为nat文件，再调用系统命令转化为BIM
    if input_file == None:
        savemat(rootpath + '\\temp.mat', {'scene': np.uint8(Img)})
        os.system(
            'importMath ' + rootpath + '/temp.mat ' + 'matlab ' + output_file + ' ' + str(img_shape[0]) + ' ' + str(
                img_shape[1]) + ' ' + str(img_shape[2]))
        os.remove(rootpath + '\\temp.mat')
    else:
        # 若有输入文件，复制其姿态信息
        savemat(rootpath + '\\temp.mat', {'scene': np.uint8(Img)})
        os.system('importMath ' + rootpath + '\\temp.mat ' + 'matlab ' + rootpath + '\\temp.BIM ' + str(
            img_shape[0]) + ' ' + str(img_shape[1]) + ' ' + str(img_shape[2]))
        os.system('ndthreshold ' + rootpath + '\\temp.BIM ' + rootpath + '\\temp2.BIM 0 1 1')
        os.system('copy_pose ' + rootpath + '\\temp2.BIM ' + input_file + ' ' + output_file)
        os.remove(rootpath + '\\temp.BIM')
        os.remove(rootpath + '\\temp2.BIM')
        os.remove(rootpath + '\\temp.mat')  # 清理临时文件


def LevelAndWidth(self):
    """计算图像的窗位窗宽"""
    scalarRange = self.reader.GetOutput().GetScalarRange()  # 计算图像标量范围
    # 过滤异常值
    print('scalarRange', scalarRange)
    if (scalarRange[0] < -1e10) | (scalarRange[0] > -1e10) | (scalarRange[1] < -1e10) | (scalarRange[1] > -1e10):
        scalarRange = (0, 4095)
    window = scalarRange[1] - scalarRange[0]  # 窗宽
    level = (scalarRange[0] + scalarRange[1]) / 5.0  # 窗位（调整为1/5范围）
    return window, level


def polar360(x_input, y_input, x_ori=0, y_ori=0):
    """极坐标转换"""
    x = x_input - x_ori  # 相对原点的x偏移
    y = y_input - y_ori  # 相对原点的y偏移
    rdius = math.hypot(y, x)  # 半径
    theta = math.degrees(math.atan2(x, y)) + (x < 0) * 360  # 角度（调整为0-360度）
    return rdius, theta


def rotation_shape(coords_list, coords_origin, rotation_angle):
    """坐标旋转"""
    rotation_coords_list = []
    for i in range(len(coords_list)):
        # 计算极坐标
        coords = coords_list[i]
        rdius, theta = polar360(coords[0], coords[1], coords_origin[0], coords_origin[1])
        # 旋转后坐标
        x_r = np.int32(coords_origin[0] + rdius * math.sin((theta + rotation_angle) / 180 * math.pi))
        y_r = np.int32(coords_origin[1] + rdius * math.cos((theta + rotation_angle) / 180 * math.pi))
        rotation_coords_list.append([x_r, y_r])

    return rotation_coords_list


def drawimplant_coordinate(drawpaper_size, drawimplant_len, drawimplant_width):
    """种植体坐标绘制"""
    coord_center = drawpaper_size // 2  # 中心点
    len_center = drawimplant_len // 2  # 长度中点
    width_center = drawimplant_width // 2  # 宽度中点
    # 植入物坐标：中心，四个角，中心，上下端点
    coords_list = [[coord_center, coord_center], [coord_center + width_center, coord_center + len_center],
                   [coord_center + width_center, coord_center - len_center],
                   [coord_center - width_center, coord_center - len_center],
                   [coord_center - width_center, coord_center + len_center],
                   [coord_center, coord_center], [coord_center, drawpaper_size], [coord_center, 0]]

    return coords_list


def MaxMin_normalization_Intensity(I, max_min_val, min_max_val):
    """最大最小值归一化"""
    Ic = np.clip(I, max_min_val, min_max_val)
    return (Ic - max_min_val) / (min_max_val - max_min_val + 1e-6)


class Ui_MainWindow(QObject):
    def setupUi(self, QMainWindow):
        """设置UI主界面
        
        初始化主窗口UI，包括设置窗口属性、创建各组件区域、
        连接信号槽以及应用响应式缩放。
        """
        # 首先初始化 QObject（必须在最开始）
        # 首先初始化 QObject（必须在最开始）
        super().__init__(QMainWindow)
        self.QMainWindow = QMainWindow
        self.subject_name = 'Subject'   #受试者名称
        self.threshold_ld = 0.5 #分割阈值
        self.outputpath = './output/'   #输出路径
        # 设置模型参数
        # 添加变量
        os.environ['VIEWNIX_ENV'] = './CAVASS/'
        current_path = os.environ['Path']
        # 将新路径添加到PATH中，并移除重复项
        os.environ['Path'] = os.pathsep.join(current_path.split(os.pathsep) + ['./CAVASS/'])
        # ---------------------------------------
        self.args = Namespace()
        self.device = torch.device("cpu")  #使用cpu
        self.args.image_size = 256
        self.args.encoder_adapter = True
        
        # 定义全局字体
        self.title_font = QFont("Microsoft YaHei", 16, QFont.Bold)  # 标题字体
        self.label_font = QFont("Microsoft YaHei", 13, QFont.Normal)  # 标签字体
        self.normal_font = QFont("Microsoft YaHei", 12, QFont.Normal)  # 常规字体
        
        if not QMainWindow.objectName():
            QMainWindow.setObjectName(u"SAMMed")
        QMainWindow.resize(1300, 900)
        QMainWindow.setWindowTitle('MedSAMate')

        # 主窗口样式 - 清爽柔和的现代医疗风格
        QMainWindow.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                border: none;
            }
            QWidget {
                font-family: "Microsoft YaHei";
            }
        """)

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(QMainWindow.sizePolicy().hasHeightForWidth())
        QMainWindow.setSizePolicy(sizePolicy)

        # 设置窗口图标 - 使用 heart_lung.ico
        try:
            QMainWindow.setWindowIcon(QIcon('./heart_lung.ico'))
        except Exception as e:
            print(f"警告：图标加载失败 - {e}")

        # 创建界面元素
        self._create_actions(QMainWindow)
        self._create_central_widget(QMainWindow)
        self._create_menubar(QMainWindow)

        # 翻译和连接槽函数
        self.retranslateUi(QMainWindow)
        QMetaObject.connectSlotsByName(QMainWindow)

        # 保存主窗口引用用于后续缩放
        self.main_window = QMainWindow
        # 安装事件过滤器监听窗口大小变化
        self.main_window.installEventFilter(self)
        # 设置初始字体大小
        self.base_font_size = 14
        self.base_icon_size = 26
        # 初始缩放（延迟执行确保按钮已创建）
        QTimer.singleShot(500, self._apply_scaling)
        # 延迟刷新布局，确保分割器正确初始化
        QTimer.singleShot(100, self._refresh_layout)

    def _refresh_layout(self):
        """刷新界面布局
        
        强制刷新布局，设置各分割器的尺寸比例，
        确保四视图正确显示并更新VTK渲染窗口。
        """
        if hasattr(self, 'splitter_main'):
            # 设置分割器比例（均等分割）
            total_width = self.splitter_main.width()
            if total_width > 0:
                self.splitter_main.setSizes([total_width // 2, total_width // 2])
            
            total_height_left = self.splitter_left.height()
            if total_height_left > 0:
                self.splitter_left.setSizes([total_height_left // 2, total_height_left // 2])
            
            total_height_right = self.splitter_right.height()
            if total_height_right > 0:
                self.splitter_right.setSizes([total_height_right // 2, total_height_right // 2])
            
            # 强制更新布局
            self.splitter_main.update()
            self.centralwidget.update()
            
            # 强制VTK渲染窗口更新
            for vtk_widget in [self.vtkWidget, self.vtkWidget2, self.vtkWidget3, self.vtkWidget4]:
                if vtk_widget:
                    vtk_widget.GetRenderWindow().Render()

    def reset_view_layout(self):
        """重置视图布局
        
        将视图布局重置为默认的2x2网格比例。
        """
        if hasattr(self, 'splitter_main'):
            # 重置分割器比例
            self.splitter_main.setSizes([1, 1])
            self.splitter_left.setSizes([1, 1])
            self.splitter_right.setSizes([1, 1])
            self._refresh_layout()
            print("视图布局已重置为默认2x2网格")

    def _create_actions(self, QMainWindow):
        """创建菜单动作
        
        创建所有QAction对象，包括文件操作、模型加载、
        分割工具、测量工具等菜单动作。
        """
        self.actionAdd_DICOM_Data = QAction(QMainWindow)
        self.actionAdd_NIFTI_Data = QAction(QMainWindow)
        self.actionAdd_IM0BIM_Data = QAction(QMainWindow)
        self.actionAdd_STL_Data = QAction(QMainWindow)
        self.actionAdd_Load_Universal_model = QAction(QMainWindow)
        self.actionAdd_Load_Lungseg_model = QAction(QMainWindow)
        self.actionAdd_Load_Custom_model = QAction(QMainWindow)
        self.actionAdd_Load_Custom_model.setObjectName("actionAdd_Load_Custom_model")
        
        # 初始化自定义模型加载器（在创建action之后）
        self.custom_model_loader = CustomModelLoader()
        self.custom_model_loader.model_loaded.connect(self._on_custom_model_loaded)
        self.custom_model_loader.model_load_failed.connect(self._on_custom_model_load_failed)
        self.current_custom_model_info = None
        
        # 创建最近使用的自定义模型子菜单
        self.menuRecentCustomModels = RecentModelsMenu(self.custom_model_loader, QMainWindow)
        self.menuRecentCustomModels.model_selected.connect(self._load_model_from_path)
        self.menuRecentCustomModels.clear_history_requested.connect(self._clear_recent_custom_models)

        # 第一段的分割相关actions
        self.pointAction = QAction(QMainWindow)
        self.pointAction.setCheckable(True)
        self.pointAction.setObjectName("pointAction")
        self.pointAction.triggered.connect(self.on_action_point)

        self.point_label_0 = QAction("label 0", QMainWindow)
        self.point_label_0.setCheckable(True)
        self.point_label_0.setChecked(False)
        self.point_label_0.triggered.connect(self.select_point_label)

        self.point_label_1 = QAction("label 1", QMainWindow)
        self.point_label_1.setCheckable(True)
        self.point_label_1.setChecked(True)
        self.point_label_1.triggered.connect(self.select_point_label)

        self.labelBoxAction = QAction(QMainWindow)
        self.labelBoxAction.setCheckable(True)
        self.labelBoxAction.setObjectName("labelBoxAction")
        self.labelBoxAction.triggered.connect(self.on_action_labelBox)

        # 创建互斥的 ActionGroup 来管理 Single 和 Multiple 选择
        self.box_label_group = QActionGroup(QMainWindow)
        self.box_label_group.setExclusive(True)

        self.box_label_single = QAction("Single", QMainWindow)
        self.box_label_single.setCheckable(True)
        self.box_label_single.setObjectName("Single")
        self.box_label_single.setChecked(True)
        self.box_label_single.triggered.connect(self.select_box_label)
        self.box_label_group.addAction(self.box_label_single)

        self.box_label_multiple = QAction("Multiple", QMainWindow)
        self.box_label_multiple.setObjectName("Multiple")
        self.box_label_multiple.setCheckable(True)
        self.box_label_multiple.setChecked(False)
        self.box_label_multiple.triggered.connect(self.select_box_label)
        self.box_label_group.addAction(self.box_label_multiple)

        # 创建Segmentation Type互斥组
        self.segmentation_type_group = QActionGroup(QMainWindow)
        self.segmentation_type_group.setExclusive(True)

        self.segmentation_type_none = QAction("None", QMainWindow)
        self.segmentation_type_none.setCheckable(True)
        self.segmentation_type_none.setChecked(True)
        self.segmentation_type_none.setObjectName("None")
        self.segmentation_type_none.triggered.connect(self.select_slice_range)
        self.segmentation_type_group.addAction(self.segmentation_type_none)

        self.segmentation_type_sliceRange = QAction("Slice Range", QMainWindow)
        self.segmentation_type_sliceRange.setCheckable(True)
        self.segmentation_type_sliceRange.setChecked(False)
        self.segmentation_type_sliceRange.setObjectName("Slice Range")
        self.segmentation_type_sliceRange.triggered.connect(self.select_slice_range)
        self.segmentation_type_group.addAction(self.segmentation_type_sliceRange)

        self.startSegmentationAction = QAction(QMainWindow)
        self.startSegmentationAction.setObjectName("startSegmentationAction")
        self.startSegmentationAction.triggered.connect(self.on_action_startSegmentation)

        self.saveResultAction = QAction(QMainWindow)
        self.saveResultAction.setObjectName("saveResultAction")
        self.saveResultAction.setCheckable(True)
        self.saveResultAction.triggered.connect(self.on_action_saveResult)

        # 工具栏actions - 保留第一段的所有连接
        self.action_ruler = QAction(QMainWindow)
        self.action_ruler.setCheckable(True)
        self.action_ruler.setObjectName("action_ruler")
        self.action_ruler.triggered.connect(self.on_action_ruler)
        self.distance_widgets_1 = []
        self.distance_widgets_2 = []
        self.distance_widgets_3 = []
        self.ruler_enable = False

        self.action_paint = QAction(QMainWindow)
        self.action_paint.setCheckable(True)
        self.action_paint.setObjectName("action_paint")
        self.action_paint.triggered.connect(self.on_action_paint)
        self.paint_enable = False

        self.action_polyline = QAction(QMainWindow)
        self.action_polyline.setCheckable(True)
        self.action_polyline.setObjectName("action_polyline")
        self.action_polyline.triggered.connect(self.on_action_polyline)
        self.polyline_enable = False

        self.action_angle = QAction(QMainWindow)
        self.action_angle.setCheckable(True)
        self.action_angle.setObjectName("action_angle")
        self.action_angle.triggered.connect(self.on_action_angle)
        self.angle_enable = False

        self.action_pixel = QAction(QMainWindow)
        self.action_pixel.setCheckable(True)
        self.action_pixel.setObjectName("action_pixel")
        self.action_pixel.triggered.connect(self.on_action_pixel)
        self.pixel_enable = False

        self.action_crosshair = QAction(QMainWindow)
        self.action_crosshair.setCheckable(True)
        self.action_crosshair.setObjectName("action_crosshair")
        self.action_crosshair.triggered.connect(self.on_action_crosshair)
        self.gps_enable = False

        self.action_reset = QAction(QMainWindow)
        self.action_reset.setObjectName("action_reset")
        self.action_reset.triggered.connect(self.on_action_reset)

        self.action_dragging_image = QAction(QMainWindow)
        self.action_dragging_image.setObjectName("action_dragging_image")
        self.action_dragging_image.setCheckable(True)
        self.action_dragging_image.triggered.connect(self.on_action_dragging_image)

        # 批量设置对象名称
        actions = [self.actionAdd_DICOM_Data, self.actionAdd_NIFTI_Data, self.actionAdd_IM0BIM_Data,
                   self.actionAdd_STL_Data, self.actionAdd_Load_Universal_model, 
                   self.actionAdd_Load_Lungseg_model, self.actionAdd_Load_Custom_model]
        for i, action in enumerate(actions):
            action.setObjectName(f"action_{i}")

        # 连接第一段的菜单触发
        self.actionAdd_DICOM_Data.triggered.connect(self.on_actionAdd_DICOM_Data)
        self.actionAdd_NIFTI_Data.triggered.connect(self.on_actionAdd_NIFTI_Data)
        self.actionAdd_IM0BIM_Data.triggered.connect(self.on_actionAdd_IM0BIM_Data)
        self.actionAdd_STL_Data.triggered.connect(self.on_actionAdd_STL_Data)
        self.actionAdd_Load_Universal_model.triggered.connect(self.on_actionAdd_Load_Universal_model)
        self.actionAdd_Load_Lungseg_model.triggered.connect(self.on_actionAdd_Load_Lungseg_model)
        self.actionAdd_Load_Custom_model.triggered.connect(self.on_actionAdd_Load_Custom_model)

        self.Flag_Load_model = False
        
        # 初始化分割类型
        self.current_segmentation_type = "None"

    def _create_central_widget(self, QMainWindow):
        """创建中心控件
        
        创建主窗口的中心控件和水平主布局，
        设置左右面板的比例关系。
        """
        self.centralwidget = QWidget(QMainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.centralwidget.setStyleSheet("background-color: #f0f0f0; border: none; margin: 0px; padding: 0px;")
        self.centralwidget.setFont(self.normal_font)

        self.horizontalLayout_6 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_6.setContentsMargins(4, 4, 4, 4)
        self.horizontalLayout_6.setSpacing(4)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(4)

        self._create_left_panel(QMainWindow)
        self._create_viewer_area(QMainWindow)

        # 设置左右面板比例 (1:3)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 3)
        self.horizontalLayout_6.addLayout(self.horizontalLayout_5)

        QMainWindow.setCentralWidget(self.centralwidget)

    def _create_left_panel(self, QMainWindow):
        """创建左侧面板
        
        创建左侧面板容器，包含主工具栏、基本信息、
        工具数据面板、对比度调整和标注面板。
        """
        # 创建左侧面板容器，设置明确背景色
        self.left_panel_widget = QWidget(self.centralwidget)
        self.left_panel_widget.setStyleSheet("background-color: #f0f0f0; border: none; margin: 0px; padding: 0px;")
        self.left_panel_widget.setFont(self.normal_font)
        
        self.verticalLayout_6 = QVBoxLayout(self.left_panel_widget)
        self.verticalLayout_6.setSpacing(6)
        self.verticalLayout_6.setContentsMargins(6, 4, 6, 6)

        self._create_main_toolbar(QMainWindow)
        self._create_basic_info(QMainWindow)
        self._create_tool_data_panel(QMainWindow)
        self._create_contrast_adjustment(QMainWindow)
        self._create_annotation_panel(QMainWindow)
        
        # 设置左侧面板各区域比例
        self.verticalLayout_6.setStretch(0, 1)  # Main Toolbar
        self.verticalLayout_6.setStretch(1, 1)  # Basic Info
        self.verticalLayout_6.setStretch(2, 3)  # Tool Data (占据更多空间)
        self.verticalLayout_6.setStretch(3, 1)  # Contrast Adjustment
        self.verticalLayout_6.setStretch(4, 0)  # Annotation (hidden by default)

        self.horizontalLayout_5.addWidget(self.left_panel_widget)

    def _create_main_toolbar(self, QMainWindow):
        """创建主工具栏
        
        创建主工具栏区域，包含标尺、画笔、折线、
        角度、像素值、十字准线和拖拽等工具按钮。
        """
        self.verticalLayout_toolbar = QVBoxLayout()
        self.verticalLayout_toolbar.setSpacing(4)

        # 标题标签 - 蓝条左边框，使用微软雅黑字体
        self.label_4 = QLabel(self.left_panel_widget)
        self.label_4.setObjectName("label_4")
        self.label_4.setFont(self.title_font)
        self.label_4.setStyleSheet("""
            QLabel {
                color: #1976d2;
                font-weight: bold;
                font-size: 18px;
                padding-left: 8px;
                border: none;
                border-left: 3px solid #2196f3;
                background-color: transparent;
                margin-top: 0px;
                margin-bottom: 2px;
            }
        """)
        self.label_4.setMinimumHeight(24)
        self.label_4.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.verticalLayout_toolbar.addWidget(self.label_4)

        # 内容容器 - QFrame
        self.frame_toolbar = QFrame(self.left_panel_widget)
        self.frame_toolbar.setObjectName("frame_toolbar")
        self.frame_toolbar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        self.frame_toolbar.setFrameShape(QFrame.StyledPanel)
        self.frame_toolbar.setFrameShadow(QFrame.Raised)

        grid_layout = QGridLayout(self.frame_toolbar)
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(8, 8, 8, 8)

        buttons = [
            ("action_ruler", "img/ruler_logo.png", 0, 0),
            ("action_paint", "img/paint.png", 0, 1),
            ("action_polyline", "img/polyline.png", 0, 2),
            ("action_angle", "img/angle.png", 0, 3),
            ("action_crosshair", "img/crosshair.png", 1, 0),
            ("action_reset", "img/reset.png", 1, 1),
            ("action_pixel", "img/view_value.png", 1, 2),
            ("action_dragging_image", "img/dragging.png", 1, 3),
        ]

        self.toolbar_buttons = []

        for btn_name, icon_path, row, col in buttons:
            btn = QPushButton(self.frame_toolbar)
            btn.setObjectName(btn_name)
            btn.setFont(self.normal_font)
            btn.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #b0b0b0;
                    border-radius: 4px;
                    padding: 2px;
                    font-family: "Microsoft YaHei";
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #e3f2fd;
                    border-color: #2196f3;
                }
                QPushButton:pressed {
                    background-color: #bbdefb;
                    border-color: #1976d2;
                }
                QPushButton:checked {
                    background-color: #2196f3;
                    border-color: #1976d2;
                    color: white;
                }
            """)
            if os.path.exists(icon_path):
                icon = QIcon()
                icon.addFile(icon_path, QSize(), QIcon.Normal, QIcon.Off)
                btn.setIcon(icon)
            btn.setIconSize(QSize(22, 22))
            grid_layout.addWidget(btn, row, col, 1, 1)
            btn.clicked.connect(getattr(self, btn_name).trigger)
            self.toolbar_buttons.append(btn)

        self.verticalLayout_toolbar.addWidget(self.frame_toolbar)
        self.verticalLayout_toolbar.setStretch(0, 0)
        self.verticalLayout_toolbar.setStretch(1, 1)
        self.verticalLayout_6.addLayout(self.verticalLayout_toolbar)
        
        # 确保工具栏背景色正确
        self.frame_toolbar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 8px;
            }
        """)

    def _create_basic_info(self, QMainWindow):
        """创建基本信息区域
        
        创建基本信息区域，包含受试者ID输入框、
        数据类型选择器和历史文件列表。
        """
        self.verticalLayout_basic = QVBoxLayout()
        self.verticalLayout_basic.setSpacing(4)

        # 标题标签 - 蓝条左边框
        self.label_5 = QLabel(self.left_panel_widget)
        self.label_5.setObjectName("label_5")
        self.label_5.setFont(self.title_font)
        self.label_5.setStyleSheet("""
            QLabel {
                color: #1976d2;
                font-weight: bold;
                font-size: 18px;
                padding-left: 8px;
                border: none;
                border-left: 3px solid #2196f3;
                background-color: transparent;
                margin-top: 4px;
                margin-bottom: 2px;
            }
        """)
        self.label_5.setMinimumHeight(24)
        self.label_5.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.verticalLayout_basic.addWidget(self.label_5)

        # 内容容器 - QFrame
        self.frame_basic = QFrame(self.left_panel_widget)
        self.frame_basic.setObjectName("frame_basic")
        self.frame_basic.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.frame_basic.setFrameShape(QFrame.StyledPanel)
        self.frame_basic.setFrameShadow(QFrame.Raised)

        # 使用QGridLayout实现紧凑布局
        self.gridLayout_basic = QGridLayout(self.frame_basic)
        self.gridLayout_basic.setSpacing(10)
        self.gridLayout_basic.setContentsMargins(10, 10, 10, 10)

        # ID行 - 标签和输入框在同一行
        self.label_Subjectname = QLabel(self.frame_basic)
        self.label_Subjectname.setObjectName("label_Subjectname")
        self.label_Subjectname.setFont(self.label_font)
        self.label_Subjectname.setStyleSheet("""
            QLabel {
                color: #333333;
                background-color: transparent;
                border: none;
                font-size: 14px;
                min-width: 36px;
                font-weight: bold;
            }
        """)
        self.gridLayout_basic.addWidget(self.label_Subjectname, 0, 0, 1, 1)

        self.lineedit_Subjectname = QLineEdit(self.frame_basic)
        self.lineedit_Subjectname.setObjectName("lineedit_Subjectname")
        self.lineedit_Subjectname.setFont(self.normal_font)
        self.lineedit_Subjectname.setText('Subject')
        self.lineedit_Subjectname.textChanged[str].connect(self.lineedit_Subjectname_change_Func)
        self.lineedit_Subjectname.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.lineedit_Subjectname.setFixedHeight(28)
        self.lineedit_Subjectname.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #b0b0b0;
                border-radius: 3px;
                padding: 4px 6px;
                font-family: "Microsoft YaHei";
                font-size: 13px;
                selection-background-color: #e3f2fd;
                selection-color: #1976d2;
            }
            QLineEdit:focus {
                border-color: #2196f3;
            }
        """)
        self.gridLayout_basic.addWidget(self.lineedit_Subjectname, 0, 1, 1, 1)

        # Type行 - 标签和下拉框在同一行
        self.label_3 = QLabel(self.frame_basic)
        self.label_3.setObjectName("label_3")
        self.label_3.setFont(self.label_font)
        self.label_3.setStyleSheet("""
            QLabel {
                color: #333333;
                background-color: transparent;
                border: none;
                font-size: 14px;
                min-width: 36px;
                font-weight: bold;
            }
        """)
        self.gridLayout_basic.addWidget(self.label_3, 1, 0, 1, 1)

        self.comboBox = QComboBox(self.frame_basic)
        self.comboBox.addItems(["DICOM", "Seg", "IM0", "STL"])
        self.comboBox.setObjectName("comboBox")
        self.comboBox.setFont(self.normal_font)
        self.comboBox.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.comboBox.setFixedHeight(28)
        self.comboBox.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #b0b0b0;
                border-radius: 3px;
                padding: 4px 6px;
                font-family: "Microsoft YaHei";
                font-size: 13px;
            }
            QComboBox:hover {
                border-color: #2196f3;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
        """)
        self.gridLayout_basic.addWidget(self.comboBox, 1, 1, 1, 1)

        self.verticalLayout_basic.addWidget(self.frame_basic)
        
        # 添加历史文件列表控件
        self.history_file_list = HistoryFileListWidget(self.left_panel_widget)
        self.verticalLayout_basic.addWidget(self.history_file_list)
        
        # 连接历史文件选择信号
        self.history_file_list.file_selected.connect(self._on_history_file_selected)
        self.history_file_list.file_remove_requested.connect(self._on_history_file_remove)
        
        # 连接 ID 和 Type 变化信号到历史文件搜索
        self.lineedit_Subjectname.textChanged.connect(self._update_history_file_search)
        self.comboBox.currentTextChanged.connect(self._update_history_file_search)
        
        self.verticalLayout_basic.setStretch(0, 0)  # 标题
        self.verticalLayout_basic.setStretch(1, 0)  # 基本信息框架
        self.verticalLayout_basic.setStretch(2, 0)  # 历史文件列表
        self.verticalLayout_6.addLayout(self.verticalLayout_basic)

    def _create_tool_data_panel(self, QMainWindow):
        """创建工具数据面板
        
        创建工具数据面板，用于显示测量、标注和分割数据信息。
        """
        # 创建信息面板（从 toolbar_data_system 导入）
        self.info_panel = InfoPanel(self.left_panel_widget)
        self.info_panel.setObjectName("info_panel")
        
        # 设置样式使其与整体风格一致
        self.info_panel.setStyleSheet("""
            QFrame#info_panel {
                background-color: #f5f5f5;
                border: none;
            }
            QLabel {
                font-family: "Microsoft YaHei";
            }
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
                font-family: "Microsoft YaHei";
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 2px 6px;
                background-color: white;
                font-size: 12px;
            }
        """)
        
        # 添加到左侧布局（在 Basic Info 和 Contrast Adjustment 之间）
        self.verticalLayout_6.addWidget(self.info_panel)
        
        # 保存数据管理器引用（单例）
        self.data_manager = ToolbarDataManager()

    def _create_contrast_adjustment(self, QMainWindow):
        """创建对比度调整区域
        
        创建对比度调整区域，包含窗位和窗宽滑块，
        用于调节医学图像的显示对比度。
        """
        self.verticalLayout_contrast = QVBoxLayout()
        self.verticalLayout_contrast.setSpacing(4)

        # 标题标签 - 蓝条左边框
        self.label_6 = QLabel(self.left_panel_widget)
        self.label_6.setObjectName("label_6")
        self.label_6.setFont(self.title_font)
        self.label_6.setStyleSheet("""
            QLabel {
                color: #1976d2;
                font-weight: bold;
                font-size: 18px;
                padding-left: 8px;
                border: none;
                border-left: 3px solid #2196f3;
                background-color: transparent;
                margin-top: 4px;
                margin-bottom: 2px;
            }
        """)
        self.label_6.setMinimumHeight(24)
        self.label_6.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.verticalLayout_contrast.addWidget(self.label_6)

        # 内容容器 - QFrame
        self.frame_contrast = QFrame(self.left_panel_widget)
        self.frame_contrast.setObjectName("frame_contrast")
        self.frame_contrast.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored))
        self.frame_contrast.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.frame_contrast.setFrameShape(QFrame.StyledPanel)
        self.frame_contrast.setFrameShadow(QFrame.Raised)
        self.frame_contrast.setMinimumHeight(120)

        slider_layout = QVBoxLayout(self.frame_contrast)
        slider_layout.setSpacing(2)
        slider_layout.setContentsMargins(6, 2, 6, 2)

        # Window Level 标签和滑块
        self.label_7 = QLabel(self.frame_contrast)
        self.label_7.setObjectName("label_7")
        self.label_7.setFont(self.label_font)
        self.label_7.setStyleSheet("""
            QLabel {
                color: #333333;
                background-color: transparent;
                border: none;
                font-size: 13px;
                padding: 0px;
                margin: 0px;
                font-weight: bold;
            }
        """)
        self.label_7.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        slider_layout.addWidget(self.label_7)

        self.window_level_slider = QSlider(self.frame_contrast)
        self.window_level_slider.setObjectName("window_level_slider")
        self.window_level_slider.setMaximum(3000)
        self.window_level_slider.setMinimum(-2000)
        self.window_level_slider.setSingleStep(1)
        self.window_level_slider.setFixedHeight(28)
        self.window_level_slider.valueChanged.connect(self.valuechange4)
        self.window_level_slider.setStyleSheet("""
            QSlider {
                border: none;
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                border: none;
                background-color: #d0d0d0;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2196f3, stop:1 #64b5f6);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #ffffff;
                border: 2px solid #2196f3;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background-color: #e3f2fd;
                border-color: #1976d2;
                border-width: 3px;
            }
        """)
        self.window_level_slider.setOrientation(Qt.Horizontal)
        slider_layout.addWidget(self.window_level_slider)

        # Window Width 标签和滑块
        self.label_8 = QLabel(self.frame_contrast)
        self.label_8.setObjectName("label_8")
        self.label_8.setFont(self.label_font)
        self.label_8.setStyleSheet("""
            QLabel {
                color: #333333;
                background-color: transparent;
                border: none;
                font-size: 13px;
                padding: 0px;
                margin: 0px;
                font-weight: bold;
            }
        """)
        self.label_8.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        slider_layout.addWidget(self.label_8)

        self.window_width_slider = QSlider(self.frame_contrast)
        self.window_width_slider.setObjectName("window_width_slider")
        self.window_width_slider.setMaximum(8000)
        self.window_width_slider.setMinimum(-2000)
        self.window_width_slider.setSingleStep(1)
        self.window_width_slider.setFixedHeight(28)
        self.window_width_slider.valueChanged.connect(self.valuechange5)
        self.window_width_slider.setStyleSheet(self.window_level_slider.styleSheet())
        self.window_width_slider.setOrientation(Qt.Horizontal)
        slider_layout.addWidget(self.window_width_slider)

        self.verticalLayout_contrast.addWidget(self.frame_contrast)
        self.verticalLayout_contrast.setStretch(0, 0)
        self.verticalLayout_contrast.setStretch(1, 1)
        self.verticalLayout_6.addLayout(self.verticalLayout_contrast)

    def _create_annotation_panel(self, QMainWindow):
        """创建标注面板
        
        创建标注操作面板，包含撤销、重做和清除按钮。
        """
        self.widget_labels = QFrame(self.left_panel_widget)
        self.widget_labels.setObjectName("widget_labels")
        self.widget_labels.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 2px solid #d0d0d0;
                border-radius: 8px;
            }
        """)
        self.widget_labels.hide()
        self.widget_labels.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.labels_vertical_layout = QVBoxLayout(self.widget_labels)
        self.labels_vertical_layout.setContentsMargins(12, 10, 12, 10)
        self.labels_vertical_layout.setSpacing(10)

        # Buttons layout
        self.pushButton_layout = QHBoxLayout()
        self.pushButton_layout.setSpacing(10)

        btn_style = """
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #bdbdbd;
                border-radius: 3px;
                padding: 8px 16px;
                color: #424242;
                font-size: 14px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
                border-color: #2196f3;
            }
            QPushButton:pressed {
                background-color: #bbdefb;
            }
        """

        self.pushButton_clear = QPushButton(self.widget_labels)
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.pushButton_clear.setFont(self.normal_font)
        self.pushButton_clear.clicked.connect(self.label_clear)
        self.pushButton_clear.setStyleSheet(btn_style)

        self.pushButton_undo = QPushButton(self.widget_labels)
        self.pushButton_undo.setObjectName("pushButton_undo")
        self.pushButton_undo.setFont(self.normal_font)
        self.pushButton_undo.clicked.connect(self.label_undo)
        self.pushButton_undo.setStyleSheet(btn_style)

        self.pushButton_redo = QPushButton(self.widget_labels)
        self.pushButton_redo.setObjectName("pushButton_redo")
        self.pushButton_redo.setFont(self.normal_font)
        self.pushButton_redo.clicked.connect(self.label_redo)
        self.pushButton_redo.setStyleSheet(btn_style)

        self.pushButton_layout.addWidget(self.pushButton_redo)
        self.pushButton_layout.addWidget(self.pushButton_undo)
        self.pushButton_layout.addWidget(self.pushButton_clear)
        self.labels_vertical_layout.addLayout(self.pushButton_layout)

        self.verticalLayout_6.addWidget(self.widget_labels)

    def _create_viewer_area(self, QMainWindow):
        """创建图像显示区域
        
        创建右侧2x2网格视图区域，包含XY轴向视图、XZ冠状面视图、
        YZ矢状面视图和3D体绘制视图，使用QSplitter实现可调整布局。
        """
        
        # 创建水平分割器（左右分割）
        self.splitter_main = QSplitter(Qt.Horizontal, self.centralwidget)
        self.splitter_main.setObjectName("splitter_main")
        self.splitter_main.setHandleWidth(4)
        self.splitter_main.setStyleSheet("""
            QSplitter::handle {
                background-color: #c0c0c0;
                border: 1px solid #a0a0a0;
            }
            QSplitter::handle:hover {
                background-color: #2196f3;
            }
        """)
        
        # 左侧垂直分割器（上：XY轴向视图，下：XZ冠状面视图）
        self.splitter_left = QSplitter(Qt.Vertical, self.splitter_main)
        self.splitter_left.setObjectName("splitter_left")
        self.splitter_left.setHandleWidth(4)
        self.splitter_left.setStyleSheet(self.splitter_main.styleSheet())
        
        # 右侧垂直分割器（上：YZ矢状面视图，下：3D体绘制视图）
        self.splitter_right = QSplitter(Qt.Vertical, self.splitter_main)
        self.splitter_right.setObjectName("splitter_right")
        self.splitter_right.setHandleWidth(4)
        self.splitter_right.setStyleSheet(self.splitter_main.styleSheet())
        
        # 将左右分割器加入主分割器
        self.splitter_main.addWidget(self.splitter_left)
        self.splitter_main.addWidget(self.splitter_right)
        
        # 设置分割器比例（均等分割）
        self.splitter_main.setSizes([500, 500])
        self.splitter_left.setSizes([400, 400])
        self.splitter_right.setSizes([400, 400])
        
        # 初始化VTK相关变量
        self.pathDicomDir = ""
        self.reader = None
        
        # ==================== XY视图 (横断面/轴向) - (0,0) ====================
        self.frame_XY = QFrame(self.splitter_left)
        self.frame_XY.setObjectName("frame_XY")
        self.frame_XY.setStyleSheet("background-color: #808080; border: 1px solid #666;")
        self.frame_XY.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.id_XY = "XY"
        
        self.frame_XY_layout = QVBoxLayout(self.frame_XY)
        self.frame_XY_layout.setSpacing(0)
        self.frame_XY_layout.setContentsMargins(0, 0, 0, 0)
        
        # XY视图内容容器（VTK + 滑块）
        self.frame_XY_content = QWidget(self.frame_XY)
        self.frame_XY_content_layout = QHBoxLayout(self.frame_XY_content)
        self.frame_XY_content_layout.setSpacing(0)
        self.frame_XY_content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame_XY_content)
        self.vtkWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget.setStyleSheet("background-color: #808080; border: none;")
        
        ren_xy = vtk.vtkRenderer()
        ren_xy.SetBackground(0, 0, 0)
        self.vtkWidget.GetRenderWindow().AddRenderer(ren_xy)
        self.viewer_XY = None
        
        self.frame_XY_content_layout.addWidget(self.vtkWidget)
        
        # XY视图滑块容器
        self.slider_container_XY = QWidget(self.frame_XY_content)
        self.slider_container_XY.setFixedWidth(20)
        self.slider_container_XY_layout = QHBoxLayout(self.slider_container_XY)
        self.slider_container_XY_layout.setContentsMargins(2, 0, 2, 0)
        
        self.verticalSlider_XY = QScrollBar(self.slider_container_XY)
        self.verticalSlider_XY.setObjectName("verticalSlider_XY")
        self.verticalSlider_XY.setStyleSheet("""
            QScrollBar:vertical {
                background: #d0d0d0;
                width: 16px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical {
                background: #2196f3;
                min-height: 60px;
                border-radius: 8px;
                border: 2px solid #1976d2;
            }
            QScrollBar::handle:vertical:hover {
                background: #1976d2;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.verticalSlider_XY.setOrientation(Qt.Vertical)
        self.verticalSlider_XY.setEnabled(False)
        self.slider_container_XY_layout.addWidget(self.verticalSlider_XY)
        self.frame_XY_content_layout.addWidget(self.slider_container_XY)
        
        self.frame_XY_layout.addWidget(self.frame_XY_content)
        
        # XY视图切片标签（显示在底部）
        self.slice_label_XY = QLabel(self.frame_XY)
        self.slice_label_XY.setObjectName("slice_label_XY")
        self.slice_label_XY.setText("Slice --/--")
        self.slice_label_XY.setAlignment(Qt.AlignCenter)
        self.slice_label_XY.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                font-family: "Microsoft YaHei";
                font-size: 12px;
                padding: 2px 8px;
                border-radius: 3px;
            }
        """)
        self.slice_label_XY.setFixedHeight(24)
        self.slice_label_XY.hide()  # 初始隐藏，加载数据后显示
        self.frame_XY_layout.addWidget(self.slice_label_XY)
        
        self.splitter_left.addWidget(self.frame_XY)
        
        # ==================== XZ视图 (冠状面) - (1,0) ====================
        self.frame_XZ = QFrame(self.splitter_left)
        self.frame_XZ.setObjectName("frame_XZ")
        self.frame_XZ.setStyleSheet("background-color: #808080; border: 1px solid #666;")
        self.frame_XZ.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.id_XZ = "XZ"
        
        self.frame_XZ_layout = QVBoxLayout(self.frame_XZ)
        self.frame_XZ_layout.setSpacing(0)
        self.frame_XZ_layout.setContentsMargins(0, 0, 0, 0)
        
        # XZ视图内容容器（VTK + 滑块）
        self.frame_XZ_content = QWidget(self.frame_XZ)
        self.frame_XZ_content_layout = QHBoxLayout(self.frame_XZ_content)
        self.frame_XZ_content_layout.setSpacing(0)
        self.frame_XZ_content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtkWidget3 = QVTKRenderWindowInteractor(self.frame_XZ_content)
        self.vtkWidget3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget3.setStyleSheet("background-color: #808080; border: none;")
        
        ren_xz = vtk.vtkRenderer()
        ren_xz.SetBackground(0, 0, 0)
        self.vtkWidget3.GetRenderWindow().AddRenderer(ren_xz)
        self.viewer_XZ = None
        
        self.frame_XZ_content_layout.addWidget(self.vtkWidget3)
        
        # XZ视图滑块容器
        self.slider_container_XZ = QWidget(self.frame_XZ_content)
        self.slider_container_XZ.setFixedWidth(20)
        self.slider_container_XZ_layout = QHBoxLayout(self.slider_container_XZ)
        self.slider_container_XZ_layout.setContentsMargins(2, 0, 2, 0)
        
        self.verticalSlider_XZ = QScrollBar(self.slider_container_XZ)
        self.verticalSlider_XZ.setObjectName("verticalSlider_XZ")
        self.verticalSlider_XZ.setStyleSheet(self.verticalSlider_XY.styleSheet())
        self.verticalSlider_XZ.setOrientation(Qt.Vertical)
        self.verticalSlider_XZ.setEnabled(False)
        self.slider_container_XZ_layout.addWidget(self.verticalSlider_XZ)
        self.frame_XZ_content_layout.addWidget(self.slider_container_XZ)
        
        self.frame_XZ_layout.addWidget(self.frame_XZ_content)
        
        # XZ视图切片标签（显示在底部）
        self.slice_label_XZ = QLabel(self.frame_XZ)
        self.slice_label_XZ.setObjectName("slice_label_XZ")
        self.slice_label_XZ.setText("Slice --/--")
        self.slice_label_XZ.setAlignment(Qt.AlignCenter)
        self.slice_label_XZ.setStyleSheet(self.slice_label_XY.styleSheet())
        self.slice_label_XZ.setFixedHeight(24)
        self.slice_label_XZ.hide()  # 初始隐藏，加载数据后显示
        self.frame_XZ_layout.addWidget(self.slice_label_XZ)
        
        self.splitter_left.addWidget(self.frame_XZ)
        
        # ==================== YZ视图 (矢状面) - (0,1) ====================
        self.frame_YZ = QFrame(self.splitter_right)
        self.frame_YZ.setObjectName("frame_YZ")
        self.frame_YZ.setStyleSheet("background-color: #808080; border: 1px solid #666;")
        self.frame_YZ.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.id_YZ = "YZ"
        
        self.frame_YZ_layout = QVBoxLayout(self.frame_YZ)
        self.frame_YZ_layout.setSpacing(0)
        self.frame_YZ_layout.setContentsMargins(0, 0, 0, 0)
        
        # YZ视图内容容器（VTK + 滑块）
        self.frame_YZ_content = QWidget(self.frame_YZ)
        self.frame_YZ_content_layout = QHBoxLayout(self.frame_YZ_content)
        self.frame_YZ_content_layout.setSpacing(0)
        self.frame_YZ_content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtkWidget2 = QVTKRenderWindowInteractor(self.frame_YZ_content)
        self.vtkWidget2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget2.setStyleSheet("background-color: #808080; border: none;")
        
        ren_yz = vtk.vtkRenderer()
        ren_yz.SetBackground(0, 0, 0)
        self.vtkWidget2.GetRenderWindow().AddRenderer(ren_yz)
        self.viewer_YZ = None
        
        self.frame_YZ_content_layout.addWidget(self.vtkWidget2)
        
        # YZ视图滑块容器
        self.slider_container_YZ = QWidget(self.frame_YZ_content)
        self.slider_container_YZ.setFixedWidth(20)
        self.slider_container_YZ_layout = QHBoxLayout(self.slider_container_YZ)
        self.slider_container_YZ_layout.setContentsMargins(2, 0, 2, 0)
        
        self.verticalSlider_YZ = QScrollBar(self.slider_container_YZ)
        self.verticalSlider_YZ.setObjectName("verticalSlider_YZ")
        self.verticalSlider_YZ.setStyleSheet(self.verticalSlider_XY.styleSheet())
        self.verticalSlider_YZ.setOrientation(Qt.Vertical)
        self.verticalSlider_YZ.setEnabled(False)
        self.slider_container_YZ_layout.addWidget(self.verticalSlider_YZ)
        self.frame_YZ_content_layout.addWidget(self.slider_container_YZ)
        
        self.frame_YZ_layout.addWidget(self.frame_YZ_content)
        
        # YZ视图切片标签（显示在底部）
        self.slice_label_YZ = QLabel(self.frame_YZ)
        self.slice_label_YZ.setObjectName("slice_label_YZ")
        self.slice_label_YZ.setText("Slice --/--")
        self.slice_label_YZ.setAlignment(Qt.AlignCenter)
        self.slice_label_YZ.setStyleSheet(self.slice_label_XY.styleSheet())
        self.slice_label_YZ.setFixedHeight(24)
        self.slice_label_YZ.hide()  # 初始隐藏，加载数据后显示
        self.frame_YZ_layout.addWidget(self.slice_label_YZ)
        
        self.splitter_right.addWidget(self.frame_YZ)
        
        # ==================== 3D体绘制视图 - (1,1) ====================
        self.frame_Volume = QFrame(self.splitter_right)
        self.frame_Volume.setObjectName("frame_Volume")
        self.frame_Volume.setStyleSheet("background-color: #808080; border: 1px solid #666;")
        self.frame_Volume.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.frame_Volume_layout = QHBoxLayout(self.frame_Volume)
        self.frame_Volume_layout.setSpacing(0)
        self.frame_Volume_layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtkWidget4 = QVTKRenderWindowInteractor(self.frame_Volume)
        self.vtkWidget4.setObjectName("vtkWidget4")
        self.vtkWidget4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtkWidget4.setStyleSheet("background-color: #808080; border: none;")
        
        self.frame_Volume_layout.addWidget(self.vtkWidget4)
        
        # Volume滑块容器
        self.slider_container_Volume = QWidget(self.frame_Volume)
        self.slider_container_Volume.setFixedWidth(20)
        self.slider_container_Volume_layout = QHBoxLayout(self.slider_container_Volume)
        self.slider_container_Volume_layout.setContentsMargins(2, 0, 2, 0)
        
        self.verticalSlider_Volume = QScrollBar(self.slider_container_Volume)
        self.verticalSlider_Volume.setObjectName("verticalSlider_Volume")
        self.verticalSlider_Volume.setStyleSheet(self.verticalSlider_XY.styleSheet())
        self.verticalSlider_Volume.setOrientation(Qt.Vertical)
        self.verticalSlider_Volume.setEnabled(False)
        self.slider_container_Volume_layout.addWidget(self.verticalSlider_Volume)
        self.frame_Volume_layout.addWidget(self.slider_container_Volume)
        
        # 获取交互器和设置渲染器
        self.iren = self.vtkWidget4.GetRenderWindow().GetInteractor()
        self.reader_stl_renderer = vtk.vtkRenderer()
        self.reader_stl_renderer.SetBackground(0.5, 0.5, 0.5)
        self.reader_stl_renderer.ResetCamera()
        self.vtkWidget4.GetRenderWindow().AddRenderer(self.reader_stl_renderer)
        
        # 设置交互样式
        self.reader_stl_iren = self.iren
        self.reader_stl_style = vtk.vtkInteractorStyleTrackballCamera()
        self.reader_stl_style.SetDefaultRenderer(self.reader_stl_renderer)
        self.iren.SetInteractorStyle(self.reader_stl_style)
        self.reader_stl_style.EnabledOn()
        
        # 添加三维坐标轴指示器
        axesActor = vtk.vtkAxesActor()
        self.axes = vtk.vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axesActor)
        self.axes.SetInteractor(self.iren)
        self.axes.EnabledOn()
        self.axes.SetEnabled(1)
        self.axes.InteractiveOff()
        
        self.vtkWidget4.GetRenderWindow().Render()
        self.splitter_right.addWidget(self.frame_Volume)
        
        # 将主分割器添加到主布局
        self.horizontalLayout_5.addWidget(self.splitter_main)
        
        # 创建切片标签（用于代码逻辑，但不显示）
        self.label_XY = QLabel(self.centralwidget)
        self.label_XY.setObjectName("label_XY")
        self.label_XY.hide()
        
        self.label_YZ = QLabel(self.centralwidget)
        self.label_YZ.setObjectName("label_YZ")
        self.label_YZ.hide()
        
        self.label_XZ = QLabel(self.centralwidget)
        self.label_XZ.setObjectName("label_XZ")
        self.label_XZ.hide()
        
        self.label_Volume = QLabel(self.centralwidget)
        self.label_Volume.setObjectName("label_Volume")
        self.label_Volume.hide()
        
        # 连接切片滑动条信号
        self.verticalSlider_XY.valueChanged.connect(self.valuechange)
        self.verticalSlider_YZ.valueChanged.connect(self.valuechange2)
        self.verticalSlider_XZ.valueChanged.connect(self.valuechange3)

    def _create_menubar(self, QMainWindow):
        """创建菜单栏
        
        创建应用程序菜单栏，包含文件、加载模型、
        Med-SAM分割等菜单及其子菜单。
        """
        # 获取主窗口的菜单栏（如果不存在则创建）
        self.menubar = QMainWindow.menuBar()
        if not self.menubar:
            self.menubar = QMenuBar(QMainWindow)
            QMainWindow.setMenuBar(self.menubar)
        
        self.menubar.setObjectName(u"menubar")
        self.menubar.setStyleSheet("""
            QMenuBar {
                background-color: #e0e0e0;
                color: #000000;
                border: 1px solid #cccccc;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: #e0e0e0;
                color: #000000;
                padding: 4px 12px;
                margin: 1px;
            }
            QMenuBar::item:selected {
                background-color: #2196f3;
                color: white;
            }
        """)

        # 创建菜单 - 使用第一段的结构
        self.fileMenu = QMenu("File", self.menubar)
        self.fileMenu.setObjectName("fileMenu")

        self.modelloadMenu = QMenu("Load Models", self.menubar)
        self.modelloadMenu.setObjectName("modelloadMenu")

        self.segmentationMenu = QMenu("Med-SAM Seg", self.menubar)
        self.segmentationMenu.setObjectName("segmentationMenu")

        # 添加菜单到菜单栏
        self.menubar.addMenu(self.fileMenu)
        self.menubar.addMenu(self.modelloadMenu)
        self.menubar.addMenu(self.segmentationMenu)
        
        # 子菜单样式
        menu_style = """
            QMenu {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 2px;
            }
            QMenu::item {
                padding: 6px 20px;
                color: #333333;
                text-align: center;
            }
            QMenu::item:selected {
                background-color: #2196f3;
                color: white;
            }
        """
        
        # File菜单 - 使用第一段的actions
        self.fileMenu.addAction(self.actionAdd_DICOM_Data)
        self.fileMenu.addAction(self.actionAdd_NIFTI_Data)
        self.fileMenu.addAction(self.actionAdd_IM0BIM_Data)
        self.fileMenu.addAction(self.actionAdd_STL_Data)

        # Load Models菜单
        self.modelloadMenu.addAction(self.actionAdd_Load_Universal_model)
        self.modelloadMenu.addAction(self.actionAdd_Load_Lungseg_model)
        self.modelloadMenu.addAction(self.actionAdd_Load_Custom_model)
        self.modelloadMenu.addMenu(self.menuRecentCustomModels)

        # Segmentation菜单 - 使用第一段的结构
        self.segmentationMenu.addAction(self.pointAction)

        # Point Label子菜单
        self.point_label = QMenu("       Point Label", self.segmentationMenu)
        self.point_label.addAction(self.point_label_0)
        self.point_label.addAction(self.point_label_1)
        self.segmentationMenu.addMenu(self.point_label)

        self.segmentationMenu.addAction(self.labelBoxAction)

        # Bounding Box子菜单
        self.box_label = QMenu(" Bounding-Box Type", self.segmentationMenu)
        self.box_label.addAction(self.box_label_single)
        self.box_label.addAction(self.box_label_multiple)
        self.segmentationMenu.addMenu(self.box_label)

        self.segmentationMenu.addAction(self.startSegmentationAction)

        # Segmentation Type子菜单
        self.segmentation_type = QMenu(" Segmentation Type", self.segmentationMenu)
        self.segmentation_type.addAction(self.segmentation_type_none)
        self.segmentation_type.addAction(self.segmentation_type_sliceRange)
        self.segmentationMenu.addMenu(self.segmentation_type)

        self.segmentationMenu.addAction(self.saveResultAction)
        
        # 应用菜单样式（在子菜单创建之后）
        for menu in [self.fileMenu, self.modelloadMenu, self.segmentationMenu, 
                     self.point_label, self.box_label, self.segmentation_type]:
            menu.setStyleSheet(menu_style)

        # 添加状态栏 - 来自第一段
        self.statusBar = QStatusBar(QMainWindow)
        self.statusBar.setObjectName("statusBar")
        QMainWindow.setStatusBar(self.statusBar)

    def retranslateUi(self, QMainWindow):
        """设置界面文本翻译
        
        设置所有UI元素的显示文本，包括窗口标题、
        菜单项、标签文本和按钮文字等。
        """
        _translate = QCoreApplication.translate
        QMainWindow.setWindowTitle(_translate("SAMMed", "MedSAMate"))

        # 菜单项文本 - 来自第一段
        self.actionAdd_DICOM_Data.setText(_translate("SAMMed", "Load DICOM"))
        self.actionAdd_NIFTI_Data.setText(_translate("SAMMed", "Load Seg"))
        self.actionAdd_IM0BIM_Data.setText(_translate("SAMMed", "Load IM0"))
        self.actionAdd_STL_Data.setText(_translate("SAMMed", "Load STL"))
        self.actionAdd_Load_Universal_model.setText(_translate("SAMMed", "Load Universal Model"))
        self.actionAdd_Load_Lungseg_model.setText(_translate("SAMMed", "Load MRI Lungseg Model"))
        self.actionAdd_Load_Custom_model.setText(_translate("SAMMed", "Load Custom Model..."))

        # 分割菜单文本
        self.pointAction.setText(_translate("SAMMed", "        Point"))
        self.point_label_0.setText(_translate("SAMMed", "Label 0"))
        self.point_label_1.setText(_translate("SAMMed", "Label 1"))
        self.labelBoxAction.setText(_translate("SAMMed", "   Bounding-Box"))
        self.box_label_single.setText(_translate("SAMMed", "Single"))
        self.box_label_multiple.setText(_translate("SAMMed", "Multiple"))
        self.segmentation_type_none.setText(_translate("SAMMed", "None"))
        self.segmentation_type_sliceRange.setText(_translate("SAMMed", "Slice Range"))
        self.startSegmentationAction.setText(_translate("SAMMed", "Start Segmentation"))
        self.saveResultAction.setText(_translate("SAMMed", "   Save Results"))

        # 工具栏按钮文本为空，只显示图标
        self.action_ruler.setText("")
        self.action_ruler.setToolTip(_translate("SAMMed", "Ruler"))
        self.action_paint.setText("")
        self.action_paint.setToolTip(_translate("SAMMed", "Paint"))
        self.action_polyline.setText("")
        self.action_polyline.setToolTip(_translate("SAMMed", "Line Annotation"))
        self.action_angle.setText("")
        self.action_angle.setToolTip(_translate("SAMMed", "Angle Measurement"))
        self.action_pixel.setText("")
        self.action_pixel.setToolTip(_translate("SAMMed", "Density"))
        self.action_reset.setText("")
        self.action_reset.setToolTip(_translate("SAMMed", "Reset"))
        self.action_crosshair.setText("")
        self.action_crosshair.setToolTip(_translate("SAMMed", "Synchronous Positioning"))
        self.action_dragging_image.setText("")
        self.action_dragging_image.setToolTip(_translate("SAMMed", "Drag"))

        # 标签文本 - sam_ui(1).py样式
        self.label_4.setText(_translate("SAMMed", "  Main Toolbar"))
        self.label_5.setText(_translate("SAMMed", "  Basic Info"))
        self.label_Subjectname.setText(_translate("SAMMed", "ID:"))
        self.label_3.setText(_translate("SAMMed", "Type:"))

        # ComboBox选项
        self.comboBox.setItemText(0, _translate("SAMMed", "DICOM"))
        self.comboBox.setItemText(1, _translate("SAMMed", "Seg"))
        self.comboBox.setItemText(2, _translate("SAMMed", "IM0"))
        self.comboBox.setItemText(3, _translate("SAMMed", "STL"))

        # 对比度调整
        self.label_6.setText(_translate("SAMMed", "  Contrast Adjustment"))
        self.label_7.setText(_translate("SAMMed", "Window Level"))
        self.label_8.setText(_translate("SAMMed", "Window Width"))

        # 标注面板
        self.pushButton_clear.setText(_translate("SAMMed", "Clear"))
        self.pushButton_undo.setText(_translate("SAMMed", "Undo"))
        self.pushButton_redo.setText(_translate("SAMMed", "Redo"))

        # 菜单标题
        self.fileMenu.setTitle(_translate("SAMMed", "&File"))
        self.modelloadMenu.setTitle(_translate("SAMMed", "&Load Models"))
        self.segmentationMenu.setTitle(_translate("SAMMed", "&Med-SAM Seg"))

    def eventFilter(self, obj, event):
        """事件过滤器 - 监听窗口大小变化"""
        if obj == self.main_window and event.type() == QEvent.Resize:
            self._apply_scaling()
        return super().eventFilter(obj, event)

    def _apply_scaling(self):
        """应用响应式缩放"""
        if not hasattr(self, 'main_window') or not hasattr(self, 'toolbar_buttons'):
            return
        
        # 计算缩放因子 (基于窗口宽度)
        window_width = self.main_window.width()
        base_width = 1300
        scale_factor = max(0.7, min(1.5, window_width / base_width))
        
        # 更新图标大小和按钮尺寸 (基于48x48的基础尺寸)
        icon_size = max(24, int(28 * scale_factor))
        btn_size = max(44, int(48 * scale_factor))
        
        for btn in self.toolbar_buttons:
            btn.setIconSize(QSize(icon_size, icon_size))
            btn.setFixedSize(btn_size, btn_size)
        
        # 计算缩放后的字体大小 - 微软雅黑字体
        title_font_size = max(16, int(18 * scale_factor))
        label_font_size = max(12, int(14 * scale_factor))
        label_height = max(22, int(26 * scale_factor))
        
        # 更新标题标签 - 微软雅黑字体
        if hasattr(self, 'label_4'):
            self.label_4.setStyleSheet(f"color: #1976d2; font-weight: bold; font-size: {title_font_size}px; padding-left: 10px; border: none; border-left: 4px solid #2196f3; background-color: transparent; font-family: 'Microsoft YaHei';")
            self.label_4.setMinimumHeight(label_height)
        if hasattr(self, 'label_5'):
            self.label_5.setStyleSheet(f"color: #1976d2; font-weight: bold; font-size: {title_font_size}px; padding-left: 10px; border: none; border-left: 4px solid #2196f3; background-color: transparent; font-family: 'Microsoft YaHei';")
            self.label_5.setMinimumHeight(label_height)
        if hasattr(self, 'label_6'):
            self.label_6.setStyleSheet(f"color: #1976d2; font-weight: bold; font-size: {title_font_size}px; padding-left: 10px; border: none; border-left: 4px solid #2196f3; background-color: transparent; font-family: 'Microsoft YaHei';")
            self.label_6.setMinimumHeight(label_height)
        
        # 更新小标签 - 微软雅黑字体
        if hasattr(self, 'label_7'):
            self.label_7.setStyleSheet(f"color: #333333; background-color: transparent; border: none; font-size: {label_font_size}px; padding: 2px 0px; font-weight: bold; font-family: 'Microsoft YaHei';")
        if hasattr(self, 'label_8'):
            self.label_8.setStyleSheet(f"color: #333333; background-color: transparent; border: none; font-size: {label_font_size}px; padding: 2px 0px; font-weight: bold; font-family: 'Microsoft YaHei';")
        if hasattr(self, 'label_Subjectname'):
            self.label_Subjectname.setStyleSheet(f"color: #333333; background-color: transparent; border: none; font-size: {label_font_size}px; min-width: 40px; font-weight: bold; font-family: 'Microsoft YaHei';")
        if hasattr(self, 'label_3'):
            self.label_3.setStyleSheet(f"color: #333333; background-color: transparent; border: none; font-size: {label_font_size}px; min-width: 40px; font-weight: bold; font-family: 'Microsoft YaHei';")
        
        # 更新按钮图标大小
        if hasattr(self, 'toolbar_buttons'):
            icon_size = int(24 * scale_factor)
            for btn in self.toolbar_buttons:
                btn.setIconSize(QSize(icon_size, icon_size))

    # 警告xinxi
    @staticmethod
    def message_dialog(title, text):
        """显示警告消息对话框
        
        Args:
            title: 对话框标题
            text: 对话框显示的文本内容
        """
        msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title, text)
        msg_box.exec_()

    def lineedit_Subjectname_change_Func(self):
        """主题名称输入框内容改变处理
        
        当用户修改主题名称输入框时，更新保存的主题名称。
        """
        self.subject_name = self.lineedit_Subjectname.text()
        print(self.subject_name)

    def implant_direction_cb_up_changed(self):
        """种植体方向(上)复选框状态改变处理
        
        当选择向上方向时，取消向下方向的选中状态。
        """
        if self.implant_direction_cb_up.isChecked():
            self.implant_direction_cb_down.setCheckState(0)

    def implant_direction_cb_down_changed(self):
        """种植体方向(下)复选框状态改变处理
        
        当选择向下方向时，取消向上方向的选中状态。
        """
        if self.implant_direction_cb_down.isChecked():
            self.implant_direction_cb_up.setCheckState(0)

    def anchor_direction_cb_up_changed(self):
        """锚点方向(上)复选框状态改变处理
        
        当选择向上方向时，取消向下方向的选中状态。
        """
        if self.anchor_direction_cb_up.isChecked():
            self.anchor_direction_cb_down.setCheckState(0)

    def anchor_direction_cb_down_changed(self):
        """锚点方向(下)复选框状态改变处理
        
        当选择向下方向时，取消向上方向的选中状态。
        """
        if self.anchor_direction_cb_down.isChecked():
            self.anchor_direction_cb_up.setCheckState(0)

    def on_action_crosshair(self):
        """十字准线定位工具切换
        
        启用或禁用十字准线定位功能（GPS模式），在三个视图（XY、YZ、XZ）中
        同步显示交叉定位线，实现三维空间定位。启用时会自动禁用其他冲突工具。
        """
        if getFileIsEmpty() == True:  # 加载是否加载文件
            print("未导入文件，不能使用十字定位功能")
            return
        if self.gps_enable == False:
            # 禁止其他冲突工具（如拖动、标注）
            if self.action_dragging_image.isChecked():
                self.action_dragging_image.setChecked(False)
                # ...清理交互器
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
    # 处理 XY 视图滑块值变化
    def valuechange(self):
        # 如果没有加载数据，直接返回
        if self.viewer_XY is None:
            return
        # 当GPS功能关闭时
        if self.gps_enable == False:
            if self.pointAction.isChecked():  # 若“点标注”功能被选中
                try:
                    # 移除XY视图中所有已存在的点演员
                    for i in getPointsActor():
                        self.viewer_XY.GetRenderer().RemoveActor(i)
                    # 重新渲染XY视图
                    self.viewer_XY.Render()
                except:
                    # 移除失败时打印错误信息
                    print('Close viewer_XY actor_paint Failed!!!')
                # 获取滑块当前值（当前索引切片）
                value = self.verticalSlider_XY.value()
                # 更新XY视图的切片位置
                self.viewer_XY.SetSlice(value)
                # 若点标注的撤销栈非空（有历史标注）
                if getPointsUndoStack() != []:
                    # 遍历撤销栈中的点，找到与当前切片匹配的点
                    for point in getPointsUndoStack():
                        if point[2] == value:
                            print("valuechange")  # 调制信息：触发点重绘
                            self.point_paints(point)  # 绘制该点到XY视图
            # 若“标签框”功能被选中
            if self.labelBoxAction.isChecked():
                # 若处于“单个框标注”模式
                if getSelectSingleBoxLabel():
                    try:
                        # 移除XY视图中所有已存在的单个边界框演员（清除旧框）
                        for i in getSingleBoundingBoxActor():
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                        self.viewer_XY.Render()  # 渲染使移除生效
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                        # 获取当前切片索引并更新XY视图
                    value = self.verticalSlider_XY.value()
                    self.viewer_XY.SetSlice(value)
                    # 若单个框的撤销栈非空
                    if getSingleUndoStack() != []:
                        # 遍历撤销栈，找到与当前切片匹配的框数据
                        for data in getSingleUndoStack():
                            if data[4] == self.verticalSlider_XY.value():
                                print("single redo")  # 调试信息：单个框重绘
                                self.actor_list = []  # 清空演员列表
                                self.drwa_single_bounding_box(data)  # 绘制该框
                                setSingleBoundingBoxActor(self.actor_list)  # 保存新演员
                    # 更新视图显示范围并渲染
                    self.viewer_XY.UpdateDisplayExtent()
                    self.viewer_XY.Render()
                # 若处于“多个框标注”模式
                else:
                    try:
                        # 移除单个框演员
                        for i in getSingleBoundingBoxActor():
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    try:
                        # 移除最后一个框演员
                        for i in getLastBoundingBoxActor():
                            self.viewer_XY.GetRenderer().RemoveActor(i)
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    try:
                        # 移除所有多个框演员并清除存储
                        for actor in getMultipleBoundingBoxActor():
                            for i in actor:
                                self.viewer_XY.GetRenderer().RemoveActor(i)
                        clearMultipleBoundingBoxActor()
                        self.viewer_XY.Render()
                    except:
                        print('Close viewer_XY actor_paint Failed!!!')
                    # 更新XY视图到当前切片
                    value = self.verticalSlider_XY.value()
                    self.viewer_XY.SetSlice(value)
                    # 若多个框的撤销栈非空
                    if getMultipleUndoStack() != []:
                        # 遍历撤销栈，找到与当前切片匹配的框数据
                        for data in getMultipleUndoStack():
                            if data[4] == self.verticalSlider_XY.value():
                                print("multiple redo")  # 调试信息： 多个框重绘
                                self.actor_list = []
                                self.drwa_single_bounding_box(data)  # 绘制该框
                                setMultipleBoundingBoxActor(self.actor_list)  # 保存新演员
                    # 更新显示范围并渲染
                    self.viewer_XY.UpdateDisplayExtent()
                    self.viewer_XY.Render()
            # 若既未选中点标注也未选中标签框
            else:
                pass  # ResliceImageView Change
                # 直接更新XY视图到当前切片
                value = self.verticalSlider_XY.value()
                self.viewer_XY.SetSlice(value)
            # 统一更新XY视图的显示范围并渲染
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            try:
                # 同步更新分割视图（seg）和DICOM视图的切片
                self.viewer_seg_xy.SetSlice(value)
                self.viewer_seg_xy.UpdateDisplayExtent()
                self.viewer_seg_xy.Render()
                self.viewer_dicom_xy.SetSlice(value)
                self.viewer_dicom_xy.UpdateDisplayExtent()
                self.viewer_dicom_xy.Render()
            except:
                pass  # 视图不存在
            # 更新XY视图的切片标签（显示当前/最大切片）
            self.label_XY.setText("Slice %d/%d" % (self.viewer_XY.GetSlice(), self.viewer_XY.GetSliceMax()))
            # 更新可见的切片标签
            self.slice_label_XY.setText("Slice %d/%d" % (self.viewer_XY.GetSlice(), self.viewer_XY.GetSliceMax()))
        # 当GPS功能开启
        else:
            # 获取当前ResliceCursor的中心坐标（列表形式，便于修改）
            center = list(self.viewer_XY.GetResliceCursor().GetCenter())
            # 获取三个滑块的当前值（对应三个视图的切片）
            value_XY = self.verticalSlider_XY.value()
            value_YZ = self.verticalSlider_YZ.value()
            value_XZ = self.verticalSlider_XZ.value()
            # 获取DICOM图像的原点和间距（用于坐标转化）
            origin = self.reader.GetOutput().GetOrigin()
            spacing = self.reader.GetOutput().GetSpacing()
            # center_X = value_YZ * spacing[0] - origin[0]
            # center_Y = value_XZ * spacing[1] - origin[1]
            # 计算当前切片对应的Z轴中心坐标（基于滑块值，间距和原点）
            center_Z = value_XY * spacing[2] - origin[2]
            # 更新中心坐标的Z值
            center[2] = center_Z
            # 设置ResliceCursor的新中心并更新
            self.viewer_XY.GetResliceCursor().SetCenter(center)
            self.viewer_XY.GetResliceCursor().Update()
            # 重新渲染三个视图的ResliceCursor
            self.viewer_XY.GetResliceCursorWidget().Render()
            self.viewer_YZ.GetResliceCursorWidget().Render()
            self.viewer_XZ.GetResliceCursorWidget().Render()
            # 更新三个视图的切片标签
            self.label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # 更新可见的切片标签
            self.slice_label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.slice_label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.slice_label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # self.resliceCursorRep_XY.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.resliceCursor)
            # 渲染所有关联的窗口（使光标更新生效）
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()

    # 处理 YZ 视图滑块值变化
    def valuechange2(self):
        # 如果没有加载数据，直接返回
        if self.viewer_YZ is None:
            return
        # 当GPS功能关闭时
        if self.gps_enable == False:
            # 获取YZ滑块当前值（切片索引）
            value = self.verticalSlider_YZ.value()
            # 更新YZ视图的切片位置
            self.viewer_YZ.SetSlice(value)
            self.viewer_YZ.UpdateDisplayExtent()
            self.viewer_YZ.Render()
            try:
                # 同步更新YZ方向的分割视图和DICOM视图
                self.viewer_seg_yz.SetSlice(value)
                self.viewer_seg_yz.UpdateDisplayExtent()
                self.viewer_seg_yz.Render()
                self.viewer_dicom_yz.SetSlice(value)
                self.viewer_dicom_yz.UpdateDisplayExtent()
                self.viewer_dicom_yz.Render()
            except:
                pass  # 视图不存在
            # 更新YZ视图的切片标签
            self.label_YZ.setText("Slice %d/%d" % (self.viewer_YZ.GetSlice(), self.viewer_YZ.GetSliceMax()))
            # 更新可见的切片标签
            self.slice_label_YZ.setText("Slice %d/%d" % (self.viewer_YZ.GetSlice(), self.viewer_YZ.GetSliceMax()))

        # 当GPS功能开启时
        else:
            # 获取当前ResliceCursor的中心坐标
            center = list(self.viewer_XY.GetResliceCursor().GetCenter())
            # 获取三个滑块的值
            value_XY = self.verticalSlider_XY.value()
            value_YZ = self.verticalSlider_YZ.value()
            value_XZ = self.verticalSlider_XZ.value()
            # 获取DICOM图像的原点和间距
            origin = self.reader.GetOutput().GetOrigin()
            spacing = self.reader.GetOutput().GetSpacing()
            # 计算当前YZ切片对应的X轴中心坐标
            center_X = value_YZ * spacing[0] - origin[0]
            # center_Y = value_XZ * spacing[1] - origin[1]
            # center_Z = value_XY * spacing[2] - origin[2]
            # 更新中心坐标的X值
            center[0] = center_X
            # 更新ResliceCursor的中心并渲染
            self.viewer_XY.GetResliceCursor().SetCenter(center)
            self.viewer_XY.GetResliceCursor().Update()
            self.viewer_XY.GetResliceCursorWidget().Render()
            self.viewer_YZ.GetResliceCursorWidget().Render()
            self.viewer_XZ.GetResliceCursorWidget().Render()
            # 更新三个视图的切片标签
            self.label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # 更新可见的切片标签
            self.slice_label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.slice_label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.slice_label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # self.resliceCursorRep_XY.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.resliceCursor)
            # 渲染所有窗口
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()

    # 处理 XZ 视图滑块值变化
    def valuechange3(self):
        # 如果没有加载数据，直接返回
        if self.viewer_XZ is None:
            return
        # 当GPS功能关闭时
        if self.gps_enable == False:
            # 获取XZ滑块当前值（切片索引）
            value = self.verticalSlider_XZ.value()
            # 更新XZ视图的切片位置
            self.viewer_XZ.SetSlice(value)
            self.viewer_XZ.UpdateDisplayExtent()
            self.viewer_XZ.Render()
            try:
                # 同步更新XZ方向的分割视图和DICOM视图
                self.viewer_seg_xz.SetSlice(value)
                self.viewer_seg_xz.UpdateDisplayExtent()
                self.viewer_seg_xz.Render()
                self.viewer_dicom_xz.SetSlice(value)
                self.viewer_dicom_xz.UpdateDisplayExtent()
                self.viewer_dicom_xz.Render()
            except:
                pass  # 视图不存在
                # 更新XZ视图的切片标签
            self.label_XZ.setText("Slice %d/%d" % (self.viewer_XZ.GetSlice(), self.viewer_XZ.GetSliceMax()))
            # 更新可见的切片标签
            self.slice_label_XZ.setText("Slice %d/%d" % (self.viewer_XZ.GetSlice(), self.viewer_XZ.GetSliceMax()))
        # 当GPS功能开启时
        else:
            # 获取当前ResliceCursor的中心坐标
            center = list(self.viewer_XY.GetResliceCursor().GetCenter())
            # 获取三个滑块的值
            value_XY = self.verticalSlider_XY.value()
            value_YZ = self.verticalSlider_YZ.value()
            value_XZ = self.verticalSlider_XZ.value()
            # 获取DICOM图像的原点和间距
            origin = self.reader.GetOutput().GetOrigin()
            spacing = self.reader.GetOutput().GetSpacing()
            # center_X = value_YZ * spacing[0] - origin[0]
            # 计算当前XZ切片对应的Y轴中心坐标
            center_Y = value_XZ * spacing[1] - origin[1]
            # center_Z = value_XY * spacing[2] - origin[2]
            # 更新中心坐标的Y值
            center[1] = center_Y
            # 更新ResliceCursor的中心并渲染
            self.viewer_XY.GetResliceCursor().SetCenter(center)
            self.viewer_XY.GetResliceCursor().Update()
            self.viewer_XY.GetResliceCursorWidget().Render()
            self.viewer_YZ.GetResliceCursorWidget().Render()
            self.viewer_XZ.GetResliceCursorWidget().Render()
            # 更新三个视图的切片标签
            self.label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # 更新可见的切片标签
            self.slice_label_XY.setText("Slice %d/%d" % (value_XY, self.viewer_XY.GetSliceMax()))
            self.slice_label_YZ.setText("Slice %d/%d" % (value_YZ, self.viewer_YZ.GetSliceMax()))
            self.slice_label_XZ.setText("Slice %d/%d" % (value_XZ, self.viewer_XZ.GetSliceMax()))
            # self.resliceCursorRep_XY.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.resliceCursor)
            # 渲染所有窗口
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget2.GetRenderWindow().Render()
            self.vtkWidget3.GetRenderWindow().Render()

    # 处理窗位滑块变化
    def valuechange4(self):
        # 若未导入文件，提示并返回
        if getFileIsEmpty() == True:
            print("未导入文件，不能修改窗位数值")
            return
        # 设置滑块的 tooltip 为当前值
        self.window_level_slider.setToolTip(str(self.window_level_slider.value()))
        self.window_width_slider.setToolTip(str(self.window_width_slider.value()))
        try:
            # 更新DICOM视图的窗位和窗宽（三个方向）
            self.viewer_dicom_xy.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_xy.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_yz.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_yz.SetColorWindow(self.window_width_slider.value())
            self.viewer_dicom_xz.SetColorLevel(self.window_level_slider.value())
            self.viewer_dicom_xz.SetColorWindow(self.window_width_slider.value())
            # 渲染DICOM视图使更新生效
            self.viewer_dicom_xy.Render()
            self.viewer_dicom_yz.Render()
            self.viewer_dicom_xz.Render()
        except:
            pass  # 若DICOM视图不存在则忽略
        # 更新主视图的窗位和窗宽（三个方向）
        try:
            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.Render()
            self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_YZ.Render()
            self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_XZ.Render()
        except:
            pass  # viewer可能未初始化
        # 更新三个视图中ResliceCursor的窗位和窗宽（如果存在ResliceCursorWidget）
        try:
            self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
        except:
            pass  # vtkImageViewer2没有GetResliceCursorWidget方法

    # 处理窗宽滑块变化
    def valuechange5(self):
        # 若未导入文件，提示并返回
        if getFileIsEmpty() == True:
            print("未导入文件，不能修改窗宽数值")
            return
        # 设置滑块的 tooltip 为当前值
        self.window_level_slider.setToolTip(str(self.window_level_slider.value()))
        self.window_width_slider.setToolTip(str(self.window_width_slider.value()))
        try:
            # 更新DICOM视图的窗位和窗宽
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
            pass  # 若DICOM视图不存在则忽略
        # 更新主视图的窗位和窗宽
        try:
            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.Render()
            self.viewer_YZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_YZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_YZ.Render()
            self.viewer_XZ.SetColorLevel(self.window_level_slider.value())
            self.viewer_XZ.SetColorWindow(self.window_width_slider.value())
            self.viewer_XZ.Render()
        except:
            pass  # viewer可能未初始化
        # 更新ResliceCursor的窗位和窗宽（如果存在ResliceCursorWidget）
        try:
            self.viewer_XY.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
            self.viewer_YZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
            self.viewer_XZ.GetResliceCursorWidget().GetResliceCursorRepresentation().SetWindowLevel(
                self.window_width_slider.value(), self.window_level_slider.value())
        except:
            pass

    # 状态切换
    def switch(self):
        # 根据getNumber()的奇偶性设置状态（偶数为False，奇数为True）
        if getNumber() % 2 == 0:
            self.state = False
        else:
            self.state = True
        # 调用setNumber()更新计数器（具体逻辑在其他地方实现）
        setNumber()

    # 加载通用模型
    def on_actionAdd_Load_Universal_model(self):
        print('Load Universal_model!')  # 调试信息：开始加载通用模型
        # path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "", "*.pth;")
        # -----------------------------------------------------
        # 若已加载过模型，则删除旧模型释放资源
        if self.Flag_Load_model == True:
            del self.model
        else:
            # 首次加载，标记为已加载
            self.Flag_Load_model = True
        # -----------------------------------------------------
        # 设置通用模型的权重路径
        self.args.sam_checkpoint = "./models/pretrained/sam-med2d_b.pth"
        # 从注册表加载模型（vit_b架构），并移动到指定设备（GPU/CPU）
        self.model = sam_model_registry["vit_b"](self.args).to(self.device)
        # 显示加载成功的对话框
        self.message_dialog('Load model', 'Load Universal_model Successfully!')
        # 记录当前模型类型为通用模型
        self.modeltype = 'Universal'
        print('Load Universal_model Successfully!')  # 调试信息：加载成功

    # 加载肺分割模型
    def on_actionAdd_Load_Lungseg_model(self):
        print('Load Lungseg_model!')  # 调试信息：开始加载肺分割模型
        # path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "", "*.pth;")
        # -----------------------------------------------------
        # 若已加载过模型，则删除旧模型
        if self.Flag_Load_model == True:
            del self.model
        else:
            # 首次加载，标记为已加载
            self.Flag_Load_model = True
        # -----------------------------------------------------
        # 设置肺分割模型的权重路径
        self.args.sam_checkpoint = "./models/pretrained/sam-med2d_refine.pth"
        # 加载模型（vit_b架构）并移动到指定设备
        self.model = sam_model_registry["vit_b"](self.args).to(self.device)
        # 显示加载成功的对话框
        self.message_dialog('Load model', 'Load Lungseg_model Successfully!')
        # 记录当前模型类型为MRI肺分割模型
        self.modeltype = 'MRI_Lungseg'
        print('Load Lungseg_model Successfully!')  # 调试信息：加载成功

    # =========================================================================
    # 自定义模型加载相关方法
    # =========================================================================
    
    def on_actionAdd_Load_Custom_model(self):
        """
        加载自定义模型 - 主入口
        打开文件选择对话框并加载用户选择的模型
        """
        print('Load Custom Model initiated')
        
        # 打开文件选择对话框
        filepath = self.custom_model_loader.select_model_file(self.centralwidget)
        
        if not filepath:
            print('User cancelled model selection')
            return
        
        # 加载模型
        self._load_model_from_path(filepath)
    
    def _load_model_from_path(self, filepath):
        """
        从指定路径加载模型
        
        Args:
            filepath: 模型权重文件路径
        """
        print(f'Loading model from: {filepath}')
        
        # 查找配置文件
        config_path = ModelConfigParser.find_config_file(filepath)
        
        if not config_path:
            # 询问用户是否创建默认配置
            reply = QMessageBox.question(
                self.centralwidget,
                "Configuration File Not Found",
                "No configuration file found for this model.\n"
                "A default config will be used.\n\n"
                "Do you want to create a config template file?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                # 保存配置模板
                default_config = create_default_config_template()
                config_dir = Path(filepath).parent
                template_path = Path("./models/configs/config_template.json")
                
                try:
                    with open(template_path, 'w', encoding='utf-8') as f:
                        f.write(default_config)
                    QMessageBox.information(
                        self.centralwidget,
                        "Template Created",
                        f"Configuration template saved to:\n{template_path}\n\n"
                        "Please edit this file and rename it to 'config.json' "
                        "before loading the model again."
                    )
                    return
                except Exception as e:
                    print(f"Failed to create template: {e}")
        
        # 显示加载进度提示
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # 加载模型
            success, message = self.custom_model_loader.load_model(
                filepath, 
                config_path=config_path,
                device=self.device
            )
            
            QApplication.restoreOverrideCursor()
            
            if success:
                self._finalize_custom_model_loading()
            else:
                QMessageBox.critical(
                    self.centralwidget,
                    "Model Load Failed",
                    message
                )
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self.centralwidget,
                "Error",
                f"An unexpected error occurred:\n{str(e)}"
            )
    
    def _finalize_custom_model_loading(self):
        """
        完成自定义模型加载后的设置
        """
        model_info = self.custom_model_loader.get_current_model_info()
        if not model_info:
            return
        
        self.current_custom_model_info = model_info
        
        # 卸载之前的模型（如果有）
        if self.Flag_Load_model:
            del self.model
        else:
            self.Flag_Load_model = True
        
        # 设置新模型
        self.args.sam_checkpoint = model_info.checkpoint_path
        self.model = self.custom_model_loader.get_current_model()
        self.modeltype = 'Custom'
        
        # 更新 Med-SAM Settings 中的参数
        self._update_med_sam_settings(model_info)
        
        # 更新 Basic Info 区域
        self._update_basic_info_with_model(model_info)
        
        # 检查与当前影像模态的兼容性
        self._check_modality_compatibility(model_info)
        
        # 显示成功消息
        self.message_dialog(
            'Load Custom Model',
            f'Model "{model_info.get_display_name()}" loaded successfully!\n'
            f'Architecture: {model_info.model_architecture.upper()}\n'
            f'Image Size: {model_info.image_size}x{model_info.image_size}'
        )
        
        print(f'Custom model loaded: {model_info.name} v{model_info.version}')
    
    def _on_custom_model_loaded(self, model_info):
        """模型加载成功的回调"""
        print(f"Signal: Custom model loaded - {model_info.name}")
    
    def _on_custom_model_load_failed(self, error_msg):
        """模型加载失败的回调"""
        print(f"Signal: Custom model load failed - {error_msg}")
    
    def _update_med_sam_settings(self, model_info):
        """
        根据加载的模型更新 Med-SAM Settings
        
        Args:
            model_info: 模型信息对象
        """
        # 更新 args 中的参数
        self.args.image_size = model_info.image_size
        
        # 如果有特定的预处理参数，可以在这里更新
        if model_info.normalization_method == 'window':
            # 设置窗宽窗位
            if model_info.window_level is not None:
                print(f"Setting window level: {model_info.window_level}")
            if model_info.window_width is not None:
                print(f"Setting window width: {model_info.window_width}")
        
        # 更新其他设置...
        print(f"Med-SAM Settings updated for {model_info.name}")
    
    def _update_basic_info_with_model(self, model_info):
        """
        在 Basic Info 区域显示自定义模型信息
        
        Args:
            model_info: 模型信息对象
        """
        # 获取当前ID文本
        current_id = self.lineedit_Subjectname.text()
        
        # 在ID后添加模型信息
        model_info_text = f"[{model_info.name} v{model_info.version}]"
        
        # 如果当前ID已经包含模型信息，替换它
        if '[' in current_id and ']' in current_id:
            base_id = current_id.split('[')[0].strip()
        else:
            base_id = current_id
        
        # 更新显示
        self.lineedit_Subjectname.setText(f"{base_id} {model_info_text}")
        
        print(f"Basic Info updated with model: {model_info.get_display_name()}")
    
    def _check_modality_compatibility(self, model_info):
        """
        检查模型与当前加载影像模态的兼容性
        
        Args:
            model_info: 模型信息对象
        """
        # 获取当前影像的模态（如果有）
        current_modality = self._get_current_image_modality()
        
        if current_modality:
            is_compatible, msg = self.custom_model_loader.verify_modality_compatibility(
                current_modality
            )
            
            if not is_compatible:
                QMessageBox.warning(
                    self.centralwidget,
                    "Modality Compatibility Warning",
                    f"{msg}\n\n"
                    "The model may not perform optimally on this type of image."
                )
            else:
                print(f"Modality compatibility check passed: {msg}")
    
    def _get_current_image_modality(self):
        """
        获取当前加载影像的模态类型
        
        Returns:
            模态类型字符串 (CT/MRI/dMRI) 或 None
        """
        # 这里需要根据实际的影像元数据来实现
        # 暂时返回None，表示无法确定模态
        if hasattr(self, 'current_image_modality'):
            return self.current_image_modality
        return None
    
    def _clear_recent_custom_models(self):
        """清空最近使用的自定义模型历史"""
        self.custom_model_loader.clear_recent_models()
        QMessageBox.information(
            self.centralwidget,
            "History Cleared",
            "Recent custom models history has been cleared."
        )
    
    def show_custom_model_info(self):
        """显示当前自定义模型的详细信息"""
        if self.current_custom_model_info:
            checkpoint_info = ModelConfigParser.extract_info_from_checkpoint(
                self.current_custom_model_info.checkpoint_path
            )
            
            dialog = ModelInfoDialog(
                self.current_custom_model_info, 
                checkpoint_info,
                self.centralwidget
            )
            dialog.exec_()
        else:
            QMessageBox.information(
                self.centralwidget,
                "No Custom Model",
                "No custom model is currently loaded."
            )

    # =========================================================================
    # 历史文件记录相关方法
    # =========================================================================
    
    def _update_history_file_search(self):
        """
        根据当前的 ID 和 Type 更新历史文件搜索
        """
        print("[_update_history_file_search] Checking history_file_list...")
        if hasattr(self, 'history_file_list') and self.history_file_list:
            subject_id = self.lineedit_Subjectname.text().strip()
            file_type = self.comboBox.currentText().strip()
            
            print(f"[_update_history_file_search] ID='{subject_id}', Type='{file_type}'")
            
            # 移除模型信息标记，获取基础ID
            if '[' in subject_id and ']' in subject_id:
                subject_id = subject_id.split('[')[0].strip()
            
            self.history_file_list.update_search(subject_id, file_type)
        else:
            print("[_update_history_file_search] history_file_list not found!")
    
    def _on_history_file_selected(self, file_path: str):
        """
        历史文件被选中时的处理
        
        Args:
            file_path: 选中的文件路径
        """
        print(f"Loading file from history: {file_path}")
        
        if not os.path.exists(file_path):
            QMessageBox.warning(
                self.centralwidget,
                "File Not Found",
                f"The file no longer exists:\n{file_path}"
            )
            return
        
        # 根据文件类型调用相应的加载方法
        file_type = self.comboBox.currentText().strip().upper()
        
        try:
            # 根据文件类型调用相应的加载方法
            if file_type == "DICOM":
                # 通过File菜单加载DICOM
                QMessageBox.information(
                    self.centralwidget,
                    "Open DICOM",
                    f"Please use File → Load DICOM to open:\n{file_path}"
                )
                
            elif file_type == "SEG":
                # 加载NIFTI分割文件
                if hasattr(self, 'reader') and self.reader:
                    self.imageblend_seg_mask(file_path)
                else:
                    QMessageBox.warning(
                        self.centralwidget,
                        "Warning",
                        "Please load DICOM data first before loading segmentation file."
                    )
                
            elif file_type == "IM0":
                # 通过File菜单加载IM0
                QMessageBox.information(
                    self.centralwidget,
                    "Open IM0",
                    f"Please use File → Load IM0 to open:\n{file_path}"
                )
                
            elif file_type == "STL":
                # 加载STL文件
                if hasattr(self, 'reader') and self.reader:
                    self._load_stl_from_path(file_path)
                else:
                    QMessageBox.warning(
                        self.centralwidget,
                        "Warning",
                        "Please load DICOM data first before loading STL file."
                    )
                
            else:
                QMessageBox.information(
                    self.centralwidget,
                    "File Type",
                    f"Please use File menu to open:\n{file_path}"
                )
                return
            
            # 更新历史记录
            subject_id = self.lineedit_Subjectname.text().strip()
            if '[' in subject_id and ']' in subject_id:
                subject_id = subject_id.split('[')[0].strip()
            
            self.history_file_list.add_file_record(file_path, subject_id, file_type)
            
            QMessageBox.information(
                self.centralwidget,
                "File Loaded",
                f"Successfully loaded:\n{os.path.basename(file_path)}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self.centralwidget,
                "Load Failed",
                f"Failed to load file:\n{str(e)}"
            )
    
    def _load_dicom_from_path(self, file_path: str):
        """
        从路径加载DICOM文件
        调用现有的on_actionAdd_DICOM_Data逻辑
        """
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            raise ValueError(f"Invalid directory: {dir_path}")
        
        # 设置目录路径
        setDirPath(dir_path)
        self.dataformat = 'DICOM'
        
        # 复用on_actionAdd_DICOM_Data中的核心加载逻辑
        # 先检查是否有dcm文件
        files = os.listdir(dir_path)
        dcm_files_exist = any(file.endswith(".dcm") for file in files)
        
        if not dcm_files_exist:
            raise ValueError("No DCM files found in directory")
        
        # 清理之前的显示
        self._clear_previous_display()
        
        # 加载DICOM
        self.pathDicomDir = dir_path
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
        
        # 创建分割目标矩阵
        self.segmentation_Result = np.zeros_like(self.dicomdata)
        
        # 更新三个视图
        self._update_dicom_viewers()
        
        # 更新体绘制
        self._update_volume_rendering(dir_path)
        
        # 启用滑块
        self.verticalSlider_XY.setEnabled(True)
        self.verticalSlider_YZ.setEnabled(True)
        self.verticalSlider_XZ.setEnabled(True)
        self.verticalSlider_Volume.setEnabled(True)
        
        setFileIsEmpty(False)
        setIsPutImplant(False)
        setIsGenerateImplant(False)
        setAnchorPointIsComplete(False)
        setIsAdjust(False)
        setDirPath(dir_path)
    
    def _clear_previous_display(self):
        """清理之前的显示内容"""
        # 清理标尺
        if hasattr(self, 'ruler_enable') and self.ruler_enable:
            try:
                for ruler1 in self.distance_widgets_1:
                    ruler1.Off()
                self.distance_widgets_1.clear()
                for ruler2 in self.distance_widgets_2:
                    ruler2.Off()
                self.distance_widgets_2.clear()
                for ruler3 in self.distance_widgets_3:
                    ruler3.Off()
                self.distance_widgets_3.clear()
                self.ruler_enable = False
            except:
                pass
        
        # 清理角度测量
        if hasattr(self, 'angle_enable') and self.angle_enable:
            try:
                self.angleWidget1.Off()
                self.angleWidget2.Off()
                self.angleWidget3.Off()
                self.angle_enable = False
            except:
                pass
    
    def _update_dicom_viewers(self):
        """更新DICOM视图"""
        # 更新横断面
        self.viewer_XY = vtk.vtkResliceImageViewer()
        self.viewer_XY.SetInputData(self.reader.GetOutput())
        self.viewer_XY.SetupInteractor(self.vtkWidget)
        self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.viewer_XY.SetSliceOrientationToXY()
        self.viewer_XY.Render()
        self.camera_XY = self.viewer_XY.GetRenderer().GetActiveCamera()
        self.camera_XY.ParallelProjectionOn()
        self.camera_XY.SetParallelScale(80)
        self.camera_XY_focalPoint = self.camera_XY.GetFocalPoint()
        self.camera_XY_position = self.camera_XY.GetPosition()
        self.viewer_XY.SliceScrollOnMouseWheelOff()
        self.viewer_XY.UpdateDisplayExtent()
        self.viewer_XY.Render()
        
        self.wheelforward1 = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
        self.wheelbackward1 = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY, self.id_XY)
        self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward1)
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward1)
        
        self.value_XY = self.viewer_XY.GetSlice()
        self.maxSlice1 = self.viewer_XY.GetSliceMax()
        self.verticalSlider_XY.setMaximum(self.maxSlice1)
        self.verticalSlider_XY.setMinimum(0)
        self.verticalSlider_XY.setSingleStep(1)
        self.verticalSlider_XY.setValue(self.value_XY)
        try:
            self.verticalSlider_XY.valueChanged.disconnect(self.valuechange)
        except:
            pass
        self.verticalSlider_XY.valueChanged.connect(self.valuechange)
        self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))
        
        # 更新矢状面
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
        
        self.wheelforward2 = MouseWheelForward(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
        self.wheelbackward2 = MouseWheelBackWard(self.viewer_YZ, self.label_YZ, self.verticalSlider_YZ, self.id_YZ)
        self.viewer_YZ_InteractorStyle = self.viewer_YZ.GetInteractorStyle()
        self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward2)
        self.viewer_YZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward2)
        
        self.maxSlice2 = self.viewer_YZ.GetSliceMax()
        self.verticalSlider_YZ.setMinimum(0)
        self.verticalSlider_YZ.setMaximum(self.maxSlice2)
        self.verticalSlider_YZ.setSingleStep(1)
        self.value_YZ = self.viewer_YZ.GetSlice()
        self.verticalSlider_YZ.setValue(self.value_YZ)
        try:
            self.verticalSlider_YZ.valueChanged.disconnect(self.valuechange2)
        except:
            pass
        self.verticalSlider_YZ.valueChanged.connect(self.valuechange2)
        self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
        
        # 更新冠状面
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
        
        transform_XZ = vtk.vtkTransform()
        transform_XZ.Translate(center0, center1, center2)
        transform_XZ.RotateY(180)
        transform_XZ.RotateZ(180)
        transform_XZ.Translate(-center0, -center1, -center2)
        self.viewer_XZ.GetImageActor().SetUserTransform(transform_XZ)
        
        self.wheelforward3 = MouseWheelForward(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
        self.wheelbackward3 = MouseWheelBackWard(self.viewer_XZ, self.label_XZ, self.verticalSlider_XZ, self.id_XZ)
        self.viewer_XZ_InteractorStyle = self.viewer_XZ.GetInteractorStyle()
        self.viewer_XZ.UpdateDisplayExtent()
        self.viewer_XZ.Render()
        self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward3)
        self.viewer_XZ_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward3)
        
        self.maxSlice3 = self.viewer_XZ.GetSliceMax()
        self.verticalSlider_XZ.setMinimum(0)
        self.verticalSlider_XZ.setMaximum(self.maxSlice3)
        self.verticalSlider_XZ.setSingleStep(1)
        self.value_XZ = self.viewer_XZ.GetSlice()
        self.verticalSlider_XZ.setValue(self.value_XZ)
        try:
            self.verticalSlider_XZ.valueChanged.disconnect(self.valuechange3)
        except:
            pass
        self.verticalSlider_XZ.valueChanged.connect(self.valuechange3)
        self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
        
        # 设置窗宽窗位
        window, level = LevelAndWidth(self)
        self.window_width_slider.setValue(int(window))
        self.window_level_slider.setValue(int(level))
        self.viewer_YZ.Render()
        self.viewer_XZ.Render()
    
    def _update_volume_rendering(self, path: str):
        """更新体绘制"""
        try:
            self.vtkWidget4, self.iren, self.renderer_volume = render_update(
                self, self.vtkWidget4, self.iren, path
            )
        except Exception as e:
            print(f'Create Volume error: {e}')
    
    def _load_im0_bim_from_path(self, file_path: str):
        """
        从路径加载IM0/BIM文件
        复用现有的 on_actionAdd_IM0BIM_Data 逻辑
        """
        # 设置文件路径
        self.IM0path = file_path
        self.dataformat = 'IM0'
        
        # 直接调用现有的加载方法
        # 由于 on_actionAdd_IM0BIM_Data 使用对话框，这里手动设置路径并执行核心逻辑
        subname = os.path.basename(file_path).split('.')[0]
        
        # 转换为DICOM
        self.save_dicompath_temp = self.outputpath + subname + '_temp/'
        if not os.path.exists(self.save_dicompath_temp):
            os.mkdir(self.save_dicompath_temp)
        else:
            for f in glob.glob(self.save_dicompath_temp + '*.dcm'):
                os.remove(f)
        
        # 执行转换（使用原始代码的 os.system 方式）
        os.system('mipg2dicom ' + file_path + ' ' + self.save_dicompath_temp)
        
        # 设置保存路径
        self.save_dicompath = self.outputpath + subname
        if not os.path.exists(self.save_dicompath):
            os.mkdir(self.save_dicompath)
        else:
            for f in glob.glob(self.save_dicompath + '*.dcm'):
                os.remove(f)
        
        # 转换DICOM格式
        dicom_files = glob.glob(self.save_dicompath_temp + '*.dcm')
        dicom_files.sort()
        number_slices = len(dicom_files)
        # 设置默认 SliceThickness，避免后续使用时报错
        self.SliceThickness = 1.0
        for slice_ in range(number_slices):
            dicom_file = pydicom.dcmread(dicom_files[slice_])
            convertNsave(dicom_file, './image_dcm/DCT0000.dcm', self.save_dicompath, slice_)
            self.SliceThickness = getattr(dicom_file, 'SliceThickness', 1.0)
        
        # 清理并重新加载
        self._clear_previous_display()
        
        # 加载转换后的DICOM
        setDirPath(self.save_dicompath)
        dicom_files = glob.glob(self.save_dicompath + '/*.dcm')
        if not dicom_files:
            raise ValueError("No DICOM files found after conversion")
        first_dicom = self.save_dicompath + '/0001.dcm' if os.path.exists(self.save_dicompath + '/0001.dcm') else dicom_files[0]
        self._load_dicom_from_path(first_dicom)
    
    def _load_stl_from_path(self, file_path: str):
        """
        从路径加载STL文件
        """
        actor_stl = self.LoadSTL(file_path)
        self.reader_stl_iren.SetInteractorStyle(self.reader_stl_style)
        self.reader_stl_renderer.AddActor(actor_stl)
        self.vtkWidget4.Render()
        self.reader_stl_renderer.ResetCamera()
        self.reader_stl_iren.Initialize()
    
    def _on_history_file_remove(self, file_path: str):
        """
        从历史记录中移除文件
        
        Args:
            file_path: 要移除的文件路径
        """
        if hasattr(self, 'history_file_list') and self.history_file_list:
            self.history_file_list.history_manager.remove_record(file_path)
    
    def record_file_to_history(self, file_path: str, file_type: str = None):
        """
        将文件记录到历史
        
        Args:
            file_path: 文件路径
            file_type: 文件类型（可选，默认使用当前comboBox的值）
        """
        if not file_path or not hasattr(self, 'history_file_list'):
            return
        
        subject_id = self.lineedit_Subjectname.text().strip()
        if '[' in subject_id and ']' in subject_id:
            subject_id = subject_id.split('[')[0].strip()
        
        if not file_type:
            file_type = self.comboBox.currentText().strip()
        
        if subject_id and file_type:
            self.history_file_list.add_file_record(file_path, subject_id, file_type)
            print(f"Recorded to history: {file_path} (ID: {subject_id}, Type: {file_type})")

    # 导入 NIFTI 数据
    def on_actionAdd_NIFTI_Data(self):
        # 打开文件对话框，选择.nii.gz格式的NIFTI文件
        path = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "", "*.nii.gz")
        # 若未选择文件（路径为空），则返回
        if path[0] == "":
            return
        # 调用imageblend_seg_mask处理选中的NIFTI文件（融合分割掩码与图像）
        self.imageblend_seg_mask(path[0])

    # 处理 NIFTI 分割数据并融合显示
    def imageblend_seg_mask(self, path, slice_index=None):
        # 创建NIFTI读取器并读取文件
        self.reader_seg = vtk.vtkNIFTIImageReader()
        self.reader_seg.SetFileName(path)
        self.reader_seg.Update()  # 执行读取
        # 获取DICOM图像和分割图像的数据对象
        dicom_image = self.reader.GetOutput()
        seg_image = self.reader_seg.GetOutput()
        # 对分割图像进行翻转处理（Z轴方向）
        flip1 = vtk.vtkImageFlip()
        flip1.SetInputData(seg_image)
        flip1.SetFilteredAxis(2)  # 翻转Z轴
        flip1.Update()
        # 再次翻转（Y轴方向）
        flip2 = vtk.vtkImageFlip()
        flip2.SetInputData(flip1.GetOutput())
        flip2.SetFilteredAxis(1)  # 翻转Y轴
        flip2.Update()
        # 确保分割结果与DICOM的几何信息一致
        change_info = vtk.vtkImageChangeInformation()
        change_info.SetInputConnection(flip2.GetOutputPort())
        change_info.SetOutputOrigin(dicom_image.GetOrigin())  # 对齐原点
        change_info.SetOutputSpacing(dicom_image.GetSpacing())  # 对齐间距
        change_info.Update()
        # 保存对齐后的分割图像
        self.seg_image_aligned = change_info.GetOutput()

        # 获取分割图像的实际数据范围/获取分割图像的像素值范围（用于颜色表配置）
        data_min, data_max = self.seg_image_aligned.GetScalarRange()
        # 创建颜色查找表（LUT），用于分割结果的可视化
        self.color_table = vtk.vtkLookupTable()
        self.color_table.SetNumberOfColors(256)  # 扩展颜色精度
        self.color_table.SetTableRange(data_min, data_max)  # 动态适配
        self.color_table.SetTableValue(0, 0.0, 0.0, 1.0, 0.0)  # 背景透明
        # 1~255值设为红色（不透明）
        for i in range(1, 256):
            self.color_table.SetTableValue(i, 1, 0, 0, 1.0)  # 红色不透明
        self.color_table.Build()  # 构建颜色表
        # 创建三个方向的DICOM视图（XY、YZ、XZ），返回查看器和相机
        self.viewer_dicom_xy, self.viewer_xy_camera = self.create_dicom_viewer(self.vtkWidget.GetRenderWindow(), "XY",
                                                                               slice_index)
        self.viewer_dicom_yz, self.viewer_yz_camera = self.create_dicom_viewer(self.vtkWidget2.GetRenderWindow(), "YZ",
                                                                               slice_index)
        self.viewer_dicom_xz, self.viewer_xz_camera = self.create_dicom_viewer(self.vtkWidget3.GetRenderWindow(), "XZ",
                                                                               slice_index)
        # 创建三个方向的DICOM视图（XY、YZ、XZ），返回查看器和相机
        self.viewer_camera_XY_focalPoint = self.viewer_xy_camera.GetFocalPoint()
        self.viewer_camera_XY_position = self.viewer_xy_camera.GetPosition()
        self.viewer_camera_YZ_focalPoint = self.viewer_yz_camera.GetFocalPoint()
        self.viewer_camera_YZ_position = self.viewer_yz_camera.GetPosition()
        self.viewer_camera_XZ_focalPoint = self.viewer_xz_camera.GetFocalPoint()
        self.viewer_camera_XZ_position = self.viewer_xz_camera.GetPosition()
        # 创建三个方向的分割视图（与DICOM视图叠加）
        self.viewer_seg_xy, self.viewer_dicom_interactor_xy = self.create_seg_viewer(self.vtkWidget.GetRenderWindow(),
                                                                                     "XY", self.viewer_dicom_xy,
                                                                                     slice_index)
        self.viewer_seg_yz, self.viewer_dicom_interactor_yz = self.create_seg_viewer(self.vtkWidget2.GetRenderWindow(),
                                                                                     "YZ", self.viewer_dicom_yz,
                                                                                     slice_index)
        self.viewer_seg_xz, self.viewer_dicom_interactor_xz = self.create_seg_viewer(self.vtkWidget3.GetRenderWindow(),
                                                                                     "XZ", self.viewer_dicom_xz,
                                                                                     slice_index)
        # 渲染所有视图使配置生效
        self.viewer_dicom_xy.Render()
        self.viewer_dicom_yz.Render()
        self.viewer_dicom_xz.Render()
        self.viewer_seg_xy.Render()
        self.viewer_seg_yz.Render()
        self.viewer_seg_xz.Render()
        
        # 记录到历史文件
        self.record_file_to_history(path, "Seg")

    # 创建 DICOM 图像查看器
    def create_dicom_viewer(self, render_window, viewer_type, slice_index):
        # 获取DICOM图像的边界（用于计算中心）
        bounds = self.reader.GetOutput().GetBounds()
        center0 = (bounds[1] + bounds[0]) / 2.0  # X轴中心
        center1 = (bounds[3] + bounds[2]) / 2.0  # Y轴中心
        center2 = (bounds[5] + bounds[4]) / 2.0  # Z轴中心
        # 创建VTK图像查看器
        viewer_dicom = vtk.vtkImageViewer2()
        viewer_dicom.SetInputData(self.reader.GetOutput())  # 关联DICOM数据
        viewer_dicom.SetRenderWindow(render_window)  # 关联渲染窗口
        viewer_dicom.UpdateDisplayExtent()  # 更新显示范围
        # 配置XY方向视图
        if viewer_type == "XY":
            # 若未指定初始切片，则默认显示中间切片
            if slice_index == None:
                viewer_dicom.SetSlice(int(self.viewer_XY.GetSliceMax() / 2))
                self.verticalSlider_XY.setValue(int(viewer_dicom.GetSliceMax() / 2))  # 同步滑块
                self.label_XY.setText(
                    "Slice %d/%d" % (self.verticalSlider_XY.value(), viewer_dicom.GetSliceMax()))  # 更新标签
            else:
                # 按指定切片索引显示
                viewer_dicom.SetSlice(slice_index)
                self.verticalSlider_XY.setValue(slice_index)
                self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), viewer_dicom.GetSliceMax()))

            viewer_dicom.SetSliceOrientationToXY()  # 设置切片方向为XY
        # 配置YZ方向视图
        if viewer_type == "YZ":
            # 默认显示中间切片
            viewer_dicom.SetSlice(int(self.viewer_YZ.GetSliceMax() / 2))
            self.verticalSlider_YZ.setValue(int(viewer_dicom.GetSliceMax() / 2))
            self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), viewer_dicom.GetSliceMax()))
            viewer_dicom.SetSliceOrientationToYZ()  # 切片方向为YZ
            # 对YZ视图应用旋转变换（调整显示方向与DICOM一致）
            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)  # 平移到中心
            transform_YZ.RotateX(180)  # X轴旋转180度
            transform_YZ.RotateZ(180)  # Z轴旋转180度
            transform_YZ.Translate(-center0, -center1, -center2)  # 平移回原位置
            viewer_dicom.GetImageActor().SetUserTransform(transform_YZ)  # 应用变换

        # 配置XZ方向视图
        if viewer_type == "XZ":
            # 默认显示中间切片
            viewer_dicom.SetSlice(int(self.viewer_XZ.GetSliceMax() / 2))
            self.verticalSlider_XZ.setValue(int(viewer_dicom.GetSliceMax() / 2))
            self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), viewer_dicom.GetSliceMax()))
            viewer_dicom.SetSliceOrientationToXZ()  # 切片方向为XZ
            # 对XZ视图应用旋转变换
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)  # Y轴旋转180度
            transform_XZ.RotateZ(180)  # Z轴旋转180度
            transform_XZ.Translate(-center0, -center1, -center2)
            viewer_dicom.GetImageActor().SetUserTransform(transform_XZ)
        # 设置初始窗宽窗位（与滑块同步）
        viewer_dicom.SetColorWindow(self.window_width_slider.value())
        viewer_dicom.SetColorLevel(self.window_level_slider.value())
        # 配置相机为平行投影（适合医学图像）
        camera = viewer_dicom.GetRenderer().GetActiveCamera()
        camera.ParallelProjectionOn()  # 开启平行投影
        camera.SetParallelScale(80)  # 设置放大倍数（参数跟某个数据有着数学关系，越小图像越大）
        # 返回查看器和相机对象
        return viewer_dicom, camera

    # 创建分割图像查看器，与 DICOM 视图叠加
    def create_seg_viewer(self, render_window, viewer_type, viewer_dicom, slice_index):
        # 获取DICOM图像边界，计算中心（用于变换）
        bounds = self.reader.GetOutput().GetBounds()
        center0 = (bounds[1] + bounds[0]) / 2.0
        center1 = (bounds[3] + bounds[2]) / 2.0
        center2 = (bounds[5] + bounds[4]) / 2.0
        # 创建分割图像查看器（图层）
        viewerLayer = vtk.vtkImageViewer2()
        viewerLayer.SetInputData(self.seg_image_aligned)  # 关联对齐后的分割图像
        viewerLayer.SetRenderWindow(render_window)  # 与DICOM视图共享渲染窗口
        # 配置XY方向的分割视图
        if viewer_type == "XY":
            viewerLayer.SetSliceOrientationToXY()  # 切片方向为XY
            # 获取XY视图的交互器
            rwi = self.viewer_XY.GetRenderWindow().GetInteractor()
            # 创建鼠标滚轮事件处理器（向前滚动切换切片）
            wheelforward = MouseWheelForward(viewer_dicom, self.label_XY, self.verticalSlider_XY,
                                             self.id_XY)
            # 创建鼠标滚轮事件处理器（向后滚动切换切片）
            wheelbackward = MouseWheelBackWard(viewer_dicom, self.label_XY, self.verticalSlider_XY,
                                               self.id_XY)
            # 设置初始切片（与DICOM视图一致）
            if slice_index == None:
                viewerLayer.SetSlice(viewer_dicom.GetSlice())
            else:
                viewerLayer.SetSlice(slice_index)
        # 配置YZ方向的分割视图
        elif viewer_type == "YZ":
            viewerLayer.SetSliceOrientationToYZ()  # 切片方向为YZ
            rwi = self.viewer_YZ.GetRenderWindow().GetInteractor()  # 获取交互器
            # 创建滚轮事件处理器
            wheelforward = MouseWheelForward(viewer_dicom, self.label_YZ, self.verticalSlider_YZ,
                                             self.id_YZ)
            wheelbackward = MouseWheelBackWard(viewer_dicom, self.label_YZ, self.verticalSlider_YZ,
                                               self.id_YZ)
            # 应用与DICOM视图相同的旋转变换（确保方向一致）
            transform_YZ = vtk.vtkTransform()
            transform_YZ.Translate(center0, center1, center2)
            transform_YZ.RotateX(180)
            transform_YZ.RotateZ(180)
            transform_YZ.Translate(-center0, -center1, -center2)
            viewerLayer.GetImageActor().SetUserTransform(transform_YZ)
            viewerLayer.SetSlice(viewer_dicom.GetSlice())  # 同步切片
        # 配置XZ方向的分割视图
        else:
            viewerLayer.SetSliceOrientationToXZ()  # 切片方向为XZ
            rwi = self.viewer_XZ.GetRenderWindow().GetInteractor()  # 获取交互器
            # 创建滚轮事件处理器
            wheelforward = MouseWheelForward(viewer_dicom, self.label_XZ, self.verticalSlider_XZ,
                                             self.id_XZ)
            wheelbackward = MouseWheelBackWard(viewer_dicom, self.label_XZ, self.verticalSlider_XZ,
                                               self.id_XZ)
            # 应用旋转变换
            transform_XZ = vtk.vtkTransform()
            transform_XZ.Translate(center0, center1, center2)
            transform_XZ.RotateY(180)
            transform_XZ.RotateZ(180)
            transform_XZ.Translate(-center0, -center1, -center2)
            viewerLayer.GetImageActor().SetUserTransform(transform_XZ)
            viewerLayer.SetSlice(viewer_dicom.GetSlice())  # 同步切片
        # 配置分割图像的显示属性
        viewerLayer.GetImageActor().SetInterpolate(False)  # 关闭插值（避免模糊）
        viewerLayer.GetImageActor().GetProperty().SetLookupTable(self.color_table)  # 应用颜色表
        viewerLayer.GetImageActor().GetProperty().SetDiffuse(0.0)  # 关闭漫反射（使颜色更鲜艳）
        viewerLayer.GetImageActor().SetPickable(False)  # 禁止拾取（避免干扰DICOM视图交互）
        # 将分割图像的演员添加到DICOM视图的渲染器（实现叠加显示）
        viewer_dicom.GetRenderer().AddActor(viewerLayer.GetImageActor())
        # 将DICOM视图与交互器关联
        viewer_dicom.SetupInteractor(rwi)
        # 获取DICOM视图的交互器样式
        viewer_dicom_interactor = viewer_dicom.GetInteractorStyle()
        # --------------------------------------------------------------------------------------
        # 为交互器添加鼠标滚轮事件观察者（处理切片切换）
        viewer_dicom_interactor.AddObserver("MouseWheelForwardEvent", wheelforward)
        viewer_dicom_interactor.AddObserver("MouseWheelBackwardEvent", wheelbackward)
        # 返回分割视图和交互器
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
        
        # 检查路径是否存在
        if not path or not os.path.exists(path):
            QMessageBox.warning(self.centralwidget, "Warning", "Selected path does not exist!")
            return
        
        # 转换为绝对路径
        path = os.path.abspath(path)

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
        # slider属性 - 先断开旧连接避免重复
        try:
            self.verticalSlider_XY.valueChanged.disconnect(self.valuechange)
        except:
            pass
        self.value_XY = self.viewer_XY.GetSlice()
        self.maxSlice1 = self.viewer_XY.GetSliceMax()
        self.verticalSlider_XY.setMaximum(self.maxSlice1)
        self.verticalSlider_XY.setMinimum(0)
        self.verticalSlider_XY.setSingleStep(1)
        self.verticalSlider_XY.setValue(self.value_XY)
        self.verticalSlider_XY.valueChanged.connect(self.valuechange)
        self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))
        # 显示并更新可见的切片标签
        self.slice_label_XY.show()
        self.slice_label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))
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
        try:
            self.verticalSlider_YZ.valueChanged.disconnect(self.valuechange2)
        except:
            pass
        self.maxSlice2 = self.viewer_YZ.GetSliceMax()
        self.verticalSlider_YZ.setMinimum(0)
        self.verticalSlider_YZ.setMaximum(self.maxSlice2)
        self.verticalSlider_YZ.setSingleStep(1)
        self.value_YZ = self.viewer_YZ.GetSlice()
        self.verticalSlider_YZ.setValue(self.value_YZ)
        self.verticalSlider_YZ.valueChanged.connect(self.valuechange2)
        self.label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
        # 显示并更新可见的切片标签
        self.slice_label_YZ.show()
        self.slice_label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
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
        try:
            self.verticalSlider_XZ.valueChanged.disconnect(self.valuechange3)
        except:
            pass
        self.maxSlice3 = self.viewer_XZ.GetSliceMax()
        self.verticalSlider_XZ.setMinimum(0)
        self.verticalSlider_XZ.setMaximum(self.maxSlice3)
        self.verticalSlider_XZ.setSingleStep(1)
        self.value_XZ = self.viewer_XZ.GetSlice()
        self.verticalSlider_XZ.setValue(self.value_XZ)
        self.verticalSlider_XZ.valueChanged.connect(self.valuechange3)
        self.label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
        # 显示并更新可见的切片标签
        self.slice_label_XZ.show()
        self.slice_label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
        window, level = LevelAndWidth(self)
        print(window)
        print(level)
        self.window_width_slider.setValue(int(window))
        self.window_level_slider.setValue(int(level))
        self.viewer_YZ.Render()
        self.viewer_XZ.Render()
        # ------------更新体绘制窗口数据-----------------------------------------
        try:
            self.vtkWidget4, self.iren, self.renderer_volume = render_update(self, self.vtkWidget4, self.iren, path)
            # render_update 已经在 volume.py 中创建了坐标轴，存储在 self.axes
            # 不需要再次创建
        except:
            print('Creat Volume error!')
        
        # 启用滑块
        self.verticalSlider_XY.setEnabled(True)
        self.verticalSlider_YZ.setEnabled(True)
        self.verticalSlider_XZ.setEnabled(True)
        self.verticalSlider_Volume.setEnabled(True)
        
        # 记录到历史文件
        self.record_file_to_history(path, "DICOM")

    def _add_axes_to_volume(self):
        """添加三维坐标轴到体绘制窗口 - 参考 volume.py 的 render_update 实现"""
        # 禁用旧的坐标轴widget（如果存在）
        if hasattr(self, 'axes') and self.axes:
            try:
                self.axes.SetEnabled(0)
            except:
                pass
        
        # 获取当前渲染窗口的interactor
        current_iren = self.vtkWidget4.GetRenderWindow().GetInteractor()
        if current_iren:
            self.iren = current_iren
        
        # 创建坐标轴actor - 与 volume.py 一致
        axesActor = vtk.vtkAxesActor()
        
        # 创建方向标记控件 - 与 volume.py 一致，使用 self.axes
        self.axes = vtk.vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axesActor)
        self.axes.SetInteractor(self.iren)
        self.axes.EnabledOn()
        self.axes.SetEnabled(1)
        self.axes.InteractiveOff()  # 禁用交互，坐标轴会跟随相机旋转

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
        # ---------------------------------------------------------------------------------
        self.save_dicompath = self.outputpath + subname
        if not os.path.exists(self.save_dicompath):
            os.mkdir(self.save_dicompath)
        else:
            for file in glob.glob(self.save_dicompath + '*.dcm'):
                os.remove(file)
        # --------------------------------------
        dicom_files = glob.glob(self.save_dicompath_temp + '*.dcm')
        dicom_files.sort()
        number_slices = len(dicom_files)
        # 设置默认 SliceThickness，避免后续使用时报错
        self.SliceThickness = 1.0
        for slice_ in range(number_slices):
            dicom_file = pydicom.dcmread(dicom_files[slice_])
            convertNsave(dicom_file, './image_dcm/DCT0000.dcm', self.save_dicompath, slice_)
            self.SliceThickness = getattr(dicom_file, 'SliceThickness', 1.0)
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
        # 显示并更新可见的切片标签
        self.slice_label_XY.show()
        self.slice_label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))
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
        # 显示并更新可见的切片标签
        self.slice_label_YZ.show()
        self.slice_label_YZ.setText("Slice %d/%d" % (self.verticalSlider_YZ.value(), self.viewer_YZ.GetSliceMax()))
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
        # 显示并更新可见的切片标签
        self.slice_label_XZ.show()
        self.slice_label_XZ.setText("Slice %d/%d" % (self.verticalSlider_XZ.value(), self.viewer_XZ.GetSliceMax()))
        window, level = LevelAndWidth(self)
        print(window)
        print(level)
        self.window_width_slider.setValue(int(window))
        self.window_level_slider.setValue(int(level))
        self.viewer_YZ.Render()
        self.viewer_XZ.Render()
        # ------------更新体绘制窗口数据-----------------------------------------
        try:
            self.vtkWidget4, self.iren, self.renderer_volume = render_update(self, self.vtkWidget4, self.iren, path)
            # render_update 已经在 volume.py 中创建了坐标轴
        except:
            print('Creat Volume error!')
        
        # 启用滑块
        self.verticalSlider_XY.setEnabled(True)
        self.verticalSlider_YZ.setEnabled(True)
        self.verticalSlider_XZ.setEnabled(True)
        self.verticalSlider_Volume.setEnabled(True)
        
        # 记录到历史文件
        if hasattr(self, 'IM0path') and self.IM0path:
            self.record_file_to_history(self.IM0path, "IM0")

    def LoadSTL(self, filename):
        # 首先创建STL reader读取文件
        reader = vtk.vtkSTLReader()  # 读取stl文件
        reader.SetFileName(filename)  # 文件名
        reader.Update()  # 确保数据被读取
        
        # 获取STL文件的bounds用于居中显示
        stl_bounds = reader.GetOutput().GetBounds()
        self.center0 = (stl_bounds[1] + stl_bounds[0]) / 2.0
        self.center1 = (stl_bounds[3] + stl_bounds[2]) / 2.0
        self.center2 = (stl_bounds[5] + stl_bounds[4]) / 2.0
        
        transform = vtk.vtkTransform()
        transform.Translate(-self.center0, -self.center1, -self.center2)  # 将模型中心移到原点

        mapper = vtk.vtkPolyDataMapper()  # 将多边形数据映射到图形基元
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.SetUserTransform(transform)
        return actor  # 表示渲染场景中的实体

    def on_actionAdd_STL_Data(self):
        print("选择STL文件")
        path = QtWidgets.QFileDialog.getOpenFileName(None, "选择STL文件", filter="*.stl")
        print(path)
        if path[0]:
            actor_stl = self.LoadSTL(path[0])

            self.reader_stl_iren.SetInteractorStyle(self.reader_stl_style)
            self.reader_stl_renderer.AddActor(actor_stl)
            self.vtkWidget4.Render()
            self.reader_stl_renderer.ResetCamera()
            self.reader_stl_iren.Initialize()
            
            # 记录到历史文件
            self.record_file_to_history(path[0], "STL")

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

            # 添加数据管理系统集成 - 测量完成时自动记录
            def on_ruler_complete(obj, event, view_name="XY"):
                """测量完成回调"""
                try:
                    rep = obj.GetDistanceRepresentation()
                    distance = rep.GetDistance()
                    # 获取测量端点坐标
                    p1 = [0, 0, 0]
                    p2 = [0, 0, 0]
                    rep.GetPoint1WorldPosition(p1)
                    rep.GetPoint2WorldPosition(p2)
                    
                    # 提交到数据管理系统
                    data_mgr = ToolbarDataManager()
                    from datetime import datetime
                    import uuid
                    
                    data = ToolDataModel(
                        id=str(uuid.uuid4())[:8],
                        data_type=DataType.MEASUREMENT,
                        tool_name=f"标尺 ({view_name})",
                        timestamp=datetime.now(),
                        value_data=distance,
                        coordinates=[(p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2])],
                        description=f"距离: {distance:.2f} mm",
                        metadata={'unit': 'mm', 'view': view_name},
                        color="#4CAF50"
                    )
                    data_mgr.add_data(data)
                    print(f"[数据管理] 标尺测量已记录: {distance:.2f} mm")
                except Exception as e:
                    print(f"[数据管理] 记录标尺数据失败: {e}")

            ruler1.AddObserver("EndInteractionEvent", lambda obj, event: on_ruler_complete(obj, event, "XY"))
            ruler2.AddObserver("EndInteractionEvent", lambda obj, event: on_ruler_complete(obj, event, "YZ"))
            ruler3.AddObserver("EndInteractionEvent", lambda obj, event: on_ruler_complete(obj, event, "XZ"))

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
            
            # 数据管理系统集成 - 提交画笔标注数据
            try:
                paint_points = getPaintPoint()
                if paint_points and len(paint_points) > 0:
                    data_mgr = ToolbarDataManager()
                    from datetime import datetime
                    import uuid
                    
                    # 将点列表转换为坐标列表
                    coordinates = [(p[0], p[1], p[2]) for p in paint_points]
                    
                    data = ToolDataModel(
                        id=str(uuid.uuid4())[:8],
                        data_type=DataType.ANNOTATION,
                        tool_name="画笔标注",
                        timestamp=datetime.now(),
                        value_data=len(paint_points),
                        coordinates=coordinates,
                        description=f"画笔轨迹，{len(paint_points)} 个点",
                        metadata={'tool': 'paint'},
                        color="#2196F3"
                    )
                    data_mgr.add_data(data)
                    print(f"[数据管理] 画笔标注已记录: {len(paint_points)} 个点")
                    
                    # 清空全局画笔点缓存
                    from globalVariables import paint_points as gp_paint_points
                    gp_paint_points.clear()
            except Exception as e:
                print(f"[数据管理] 记录画笔数据失败: {e}")
            
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
            
            # 数据管理系统集成 - 提交多边形标注数据
            try:
                poly_points = getPaintPoint()  # 多边形也使用 paint_points 存储
                if poly_points and len(poly_points) > 0:
                    data_mgr = ToolbarDataManager()
                    from datetime import datetime
                    import uuid
                    
                    # 将点列表转换为坐标列表
                    coordinates = [(p[0], p[1], p[2]) for p in poly_points]
                    
                    data = ToolDataModel(
                        id=str(uuid.uuid4())[:8],
                        data_type=DataType.ANNOTATION,
                        tool_name="多边形标注",
                        timestamp=datetime.now(),
                        value_data=len(poly_points),
                        coordinates=coordinates,
                        description=f"多边形标注，{len(poly_points)} 个顶点",
                        metadata={'tool': 'polyline', 'closed': True},
                        color="#9C27B0"
                    )
                    data_mgr.add_data(data)
                    print(f"[数据管理] 多边形标注已记录: {len(poly_points)} 个顶点")
                    
                    # 清空全局点缓存
                    from globalVariables import paint_points as gp_paint_points
                    gp_paint_points.clear()
            except Exception as e:
                print(f"[数据管理] 记录多边形数据失败: {e}")
            
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
            
            # 添加点击事件记录骨密度数据
            def on_pixel_click(obj, event, view_name="XY"):
                """点击记录骨密度数据"""
                try:
                    picker = vtk.vtkPointPicker()
                    picker.AddPickList(self.viewer_XY.GetImageActor())
                    iren = self.viewer_XY.GetRenderWindow().GetInteractor()
                    render = self.viewer_XY.GetRenderer()
                    pos = iren.GetEventPosition()
                    picker.Pick(pos[0], pos[1], 0, render)
                    pick_pos = picker.GetPickPosition()
                    
                    image = self.viewer_XY.GetInput()
                    origin = image.GetOrigin()
                    spacing = image.GetSpacing()
                    pixel_pos = [int(pick_pos[i] - origin[i]) / spacing[i] for i in range(3)]
                    pixel_pos = np.int32(pixel_pos)
                    
                    if (pixel_pos[0] >= 0) and (pixel_pos[1] >= 0) and (pixel_pos[2] >= 0):
                        try:
                            pixel_value = dicomdata_XY[pixel_pos[0], pixel_pos[1], pixel_pos[2]]
                            
                            # 提交到数据管理系统
                            data_mgr = ToolbarDataManager()
                            from datetime import datetime
                            import uuid
                            
                            data = ToolDataModel(
                                id=str(uuid.uuid4())[:8],
                                data_type=DataType.MEASUREMENT,
                                tool_name=f"骨密度 ({view_name})",
                                timestamp=datetime.now(),
                                value_data=float(pixel_value),
                                coordinates=[(float(pick_pos[0]), float(pick_pos[1]), float(pick_pos[2]))],
                                description=f"像素值: {pixel_value:.2f}",
                                metadata={'unit': 'HU', 'pixel_index': pixel_pos.tolist()},
                                color="#9C27B0"
                            )
                            data_mgr.add_data(data)
                            print(f"[数据管理] 骨密度已记录: {pixel_value:.2f} HU")
                        except:
                            pass
                except Exception as e:
                    print(f"[数据管理] 记录骨密度失败: {e}")
            
            self.pixel_click_observer = on_pixel_click
            self.imagestyle1.AddObserver("LeftButtonPressEvent", lambda obj, event: on_pixel_click(obj, event, "XY"))
            
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
            
            # 添加YZ视图点击事件
            def on_pixel_click_yz(obj, event):
                try:
                    picker = vtk.vtkPointPicker()
                    picker.AddPickList(self.viewer_YZ.GetImageActor())
                    iren = self.viewer_YZ.GetRenderWindow().GetInteractor()
                    render = self.viewer_YZ.GetRenderer()
                    pos = iren.GetEventPosition()
                    picker.Pick(pos[0], pos[1], 0, render)
                    pick_pos = picker.GetPickPosition()
                    
                    image = self.viewer_YZ.GetInput()
                    origin = image.GetOrigin()
                    spacing = image.GetSpacing()
                    pixel_pos = [int(pick_pos[i] - origin[i]) / spacing[i] for i in range(3)]
                    pixel_pos = np.int32(pixel_pos)
                    
                    if (pixel_pos[0] >= 0) and (pixel_pos[1] >= 0) and (pixel_pos[2] >= 0):
                        try:
                            pixel_value = dicomdata_YZ[pixel_pos[0], pixel_pos[1], pixel_pos[2]]
                            
                            data_mgr = ToolbarDataManager()
                            from datetime import datetime
                            import uuid
                            
                            data = ToolDataModel(
                                id=str(uuid.uuid4())[:8],
                                data_type=DataType.MEASUREMENT,
                                tool_name="骨密度 (YZ)",
                                timestamp=datetime.now(),
                                value_data=float(pixel_value),
                                coordinates=[(float(pick_pos[0]), float(pick_pos[1]), float(pick_pos[2]))],
                                description=f"像素值: {pixel_value:.2f}",
                                metadata={'unit': 'HU', 'pixel_index': pixel_pos.tolist()},
                                color="#9C27B0"
                            )
                            data_mgr.add_data(data)
                            print(f"[数据管理] 骨密度已记录: {pixel_value:.2f} HU")
                        except:
                            pass
                except Exception as e:
                    print(f"[数据管理] 记录骨密度失败: {e}")
            
            self.imagestyle2.AddObserver("LeftButtonPressEvent", on_pixel_click_yz)

            try:
                observer = CallBack(self.viewer_dicom_yz, self.picker2, self.vtkWidget2, dicomdata_YZ)
                self.viewer_dicom_interactor_yz.AddObserver("MouseMoveEvent", observer)
            except:
                print("don't exist")

            self.picker3 = vtk.vtkPointPicker()
            self.observer3 = CallBack(self.viewer_XZ, self.picker3, self.vtkWidget3, self.dicomdata)
            self.imagestyle3 = self.viewer_XZ.GetInteractorStyle()
            self.imagestyle3.AddObserver("MouseMoveEvent", self.observer3)
            
            # 添加XZ视图点击事件
            def on_pixel_click_xz(obj, event):
                try:
                    picker = vtk.vtkPointPicker()
                    picker.AddPickList(self.viewer_XZ.GetImageActor())
                    iren = self.viewer_XZ.GetRenderWindow().GetInteractor()
                    render = self.viewer_XZ.GetRenderer()
                    pos = iren.GetEventPosition()
                    picker.Pick(pos[0], pos[1], 0, render)
                    pick_pos = picker.GetPickPosition()
                    
                    image = self.viewer_XZ.GetInput()
                    origin = image.GetOrigin()
                    spacing = image.GetSpacing()
                    pixel_pos = [int(pick_pos[i] - origin[i]) / spacing[i] for i in range(3)]
                    pixel_pos = np.int32(pixel_pos)
                    
                    if (pixel_pos[0] >= 0) and (pixel_pos[1] >= 0) and (pixel_pos[2] >= 0):
                        try:
                            pixel_value = self.dicomdata[pixel_pos[0], pixel_pos[1], pixel_pos[2]]
                            
                            data_mgr = ToolbarDataManager()
                            from datetime import datetime
                            import uuid
                            
                            data = ToolDataModel(
                                id=str(uuid.uuid4())[:8],
                                data_type=DataType.MEASUREMENT,
                                tool_name="骨密度 (XZ)",
                                timestamp=datetime.now(),
                                value_data=float(pixel_value),
                                coordinates=[(float(pick_pos[0]), float(pick_pos[1]), float(pick_pos[2]))],
                                description=f"像素值: {pixel_value:.2f}",
                                metadata={'unit': 'HU', 'pixel_index': pixel_pos.tolist()},
                                color="#9C27B0"
                            )
                            data_mgr.add_data(data)
                            print(f"[数据管理] 骨密度已记录: {pixel_value:.2f} HU")
                        except:
                            pass
                except Exception as e:
                    print(f"[数据管理] 记录骨密度失败: {e}")
            
            self.imagestyle3.AddObserver("LeftButtonPressEvent", on_pixel_click_xz)
            
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

            # 添加数据管理系统集成 - 角度测量完成时自动记录
            def on_angle_complete(obj, event, view_name="XY"):
                """角度测量完成回调"""
                try:
                    rep = obj.GetAngleRepresentation()
                    angle = rep.GetAngle()
                    # 获取三个点坐标
                    p1, p2, p3 = [0, 0, 0], [0, 0, 0], [0, 0, 0]
                    rep.GetPoint1WorldPosition(p1)
                    rep.GetCenterWorldPosition(p2)
                    rep.GetPoint2WorldPosition(p3)
                    
                    # 提交到数据管理系统
                    data_mgr = ToolbarDataManager()
                    from datetime import datetime
                    import uuid
                    
                    data = ToolDataModel(
                        id=str(uuid.uuid4())[:8],
                        data_type=DataType.MEASUREMENT,
                        tool_name=f"角度测量 ({view_name})",
                        timestamp=datetime.now(),
                        value_data=angle,
                        coordinates=[(p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), (p3[0], p3[1], p3[2])],
                        description=f"角度: {angle:.2f}°",
                        metadata={'unit': 'degree', 'view': view_name},
                        color="#FF9800"
                    )
                    data_mgr.add_data(data)
                    print(f"[数据管理] 角度测量已记录: {angle:.2f}°")
                except Exception as e:
                    print(f"[数据管理] 记录角度数据失败: {e}")

            self.angleWidget1.AddObserver("EndInteractionEvent", lambda obj, event: on_angle_complete(obj, event, "XY"))
            self.angleWidget2.AddObserver("EndInteractionEvent", lambda obj, event: on_angle_complete(obj, event, "YZ"))
            self.angleWidget3.AddObserver("EndInteractionEvent", lambda obj, event: on_angle_complete(obj, event, "XZ"))

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
        if self.sender() == self.point_label_0 and self.point_label_0.isChecked():
            setSelectPointLabel1(False)
            self.point_label_1.setChecked(False)
            print("select point label 0")
        elif self.sender() == self.point_label_1 and self.point_label_1.isChecked():
            setSelectPointLabel1(True)
            self.point_label_0.setChecked(False)
            print("select point label 1")

    def select_box_label(self):
        # 判断是哪个 action 触发了该函数
        sender = self.sender()
        if sender == self.box_label_single:
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
        elif sender == self.box_label_multiple:
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
        if self.segmentation_type_none.isChecked():
            self.current_segmentation_type = "None"
        elif self.segmentation_type_sliceRange.isChecked():
            self.current_segmentation_type = "Slice Range"

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
        if self.modeltype in ['Universal', 'Custom']:
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
            if self.current_segmentation_type == "Slice Range":
                if self.dataformat == 'DICOM':
                    # DICOM模式：在DICOM索引空间中保持连续性
                    dicom_indices = {k: self.dicomdata.shape[2] - int(k) - 1 for k in points_dict.keys()}
                    dicom_keys = sorted(dicom_indices.values())
                    print("dicom_keys:", dicom_keys)
                    # 在DICOM索引空间填充中间切片
                    for dicom_z in range(dicom_keys[0] + 1, dicom_keys[-1]):
                        ui_index = str(self.dicomdata.shape[2] - dicom_z - 1)
                        if ui_index not in points_dict:
                            # 找到最近的已标注切片（在DICOM空间中）
                            closest_ui = min(dicom_indices.keys(),
                                             key=lambda k: abs(dicom_indices[k] - dicom_z))
                            points_dict[ui_index] = {
                                'points': points_dict[closest_ui]["points"],
                                'label': points_dict[closest_ui]['label'],
                                'image_name': f'_image_{ui_index}.png'
                            }
                else:
                    # 非DICOM模式：保持原有逻辑
                    keys = sorted(int(k) for k in points_dict.keys())
                    print("keys:", keys)
                    for i in range(keys[0] + 1, keys[-1]):
                        if str(i) not in points_dict:
                            points_dict[str(i)] = {'points': points_dict[str(i - 1)]["points"],
                                                   'label': points_dict[str(i - 1)]['label'],
                                                   'image_name': f'_image_{i}.png'}
            print(points_dict)
            for index, data in points_dict.items():
                # IM0格式也需要索引转换，因为显示时会翻转Z轴
                if self.dataformat == 'DICOM' or self.dataformat == 'IM0':
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
                if self.current_segmentation_type == "Slice Range":
                    if self.dataformat == 'DICOM':
                        # DICOM模式：在DICOM索引空间中保持连续性
                        dicom_indices = {k: self.dicomdata.shape[2] - int(k) - 1 for k in bounding_box_dict.keys()}
                        dicom_keys = sorted(dicom_indices.values())
                        print("dicom_keys:", dicom_keys)
                        # 在DICOM索引空间填充中间切片
                        for dicom_z in range(dicom_keys[0] + 1, dicom_keys[-1]):
                            ui_index = str(self.dicomdata.shape[2] - dicom_z - 1)
                            if ui_index not in bounding_box_dict:
                                # 找到最近的已标注切片（在DICOM空间中）
                                closest_ui = min(dicom_indices.keys(),
                                                 key=lambda k: abs(dicom_indices[k] - dicom_z))
                                bounding_box_dict[ui_index] = {
                                    'bounding_box': bounding_box_dict[closest_ui]["bounding_box"],
                                    'image_name': f'_image_{ui_index}.png'
                                }
                    else:
                        # 非DICOM模式：保持原有逻辑
                        keys = sorted(int(k) for k in bounding_box_dict.keys())
                        print("keys:", keys)
                        for i in range(keys[0] + 1, keys[-1]):
                            if str(i) not in bounding_box_dict:
                                bounding_box_dict[str(i)] = {'bounding_box': bounding_box_dict[str(i - 1)]["bounding_box"],
                                                             'image_name': f'_image_{i}.png'}
                print(bounding_box_dict)
                for index, data in bounding_box_dict.items():
                    # IM0格式也需要索引转换，因为显示时会翻转Z轴
                    if self.dataformat == 'DICOM' or self.dataformat == 'IM0':
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
                print(f"DEBUG: bounding_box_dict after filter={bounding_box_dict}")
                print(f"DEBUG: current_segmentation_type={self.current_segmentation_type}")
                if self.current_segmentation_type == "Slice Range":
                    print("="*50)
                    print("修复代码正在执行!")
                    print("="*50)
                    print(f"DEBUG: dataformat={self.dataformat}, shape={self.dicomdata.shape}")
                    if self.dataformat == 'DICOM':
                        # DICOM模式：在DICOM索引空间中保持连续性
                        dicom_indices = {k: self.dicomdata.shape[2] - int(k) - 1 for k in bounding_box_dict.keys()}
                        dicom_keys = sorted(dicom_indices.values())
                        print(f"DEBUG: dicom_indices={dicom_indices}, dicom_keys={dicom_keys}")
                        # 在DICOM索引空间填充中间切片
                        for dicom_z in range(dicom_keys[0] + 1, dicom_keys[-1]):
                            ui_index = str(self.dicomdata.shape[2] - dicom_z - 1)
                            print(f"DEBUG: filling dicom_z={dicom_z} -> ui_index={ui_index}")
                            if ui_index not in bounding_box_dict:
                                # 找到最近的已标注切片（在DICOM空间中）
                                closest_ui = min(dicom_indices.keys(),
                                                 key=lambda k: abs(dicom_indices[k] - dicom_z))
                                bounding_box_dict[ui_index] = {
                                    'bounding_box': bounding_box_dict[closest_ui]["bounding_box"],
                                    'image_name': f'_image_{ui_index}.png'
                                }
                                print(f"DEBUG: filled {ui_index} from {closest_ui}")
                    else:
                        # 非DICOM模式：保持原有逻辑
                        keys = sorted(int(k) for k in bounding_box_dict.keys())
                        print("keys:", keys)
                        for i in range(keys[0] + 1, keys[-1]):
                            if str(i) not in bounding_box_dict:
                                bounding_box_dict[str(i)] = {'bounding_box': bounding_box_dict[str(i - 1)]["bounding_box"],
                                                             'image_name': f'_image_{i}.png'}
                print(f"DEBUG: final bounding_box_dict={bounding_box_dict}")
                for index, data in bounding_box_dict.items():
                    # IM0格式也需要索引转换，因为显示时会翻转Z轴
                    if self.dataformat == 'DICOM' or self.dataformat == 'IM0':
                        index_z = self.dicomdata.shape[2] - int(index) - 1
                    else:
                        index_z = int(index)
                    print(f"DEBUG: processing index={index}, index_z={index_z}")
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
                    print(f"DEBUG: saving to segmentation_Result[:, :, {index_z}]")
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
        """保存分割结果"""
        try:
            # 检查分割结果是否存在
            if not hasattr(self, 'segmentation_Result'):
                QMessageBox.warning(self.QMainWindow, "Save Failed", "Please perform segmentation first!")
                return
            
            Segmentation_Result = self.segmentation_Result
            
            # 检查分割结果是否为空
            if Segmentation_Result is None or np.sum(Segmentation_Result) == 0:
                QMessageBox.warning(self.QMainWindow, "Save Failed", "Segmentation result is empty!")
                return
            
            # 确保输出目录存在
            if not os.path.exists(self.outputpath):
                os.makedirs(self.outputpath)
            
            saved_files = []
            
            # 保存分割结果
            if self.dataformat == 'IM0':
                Segmentation_Result = np.transpose(Segmentation_Result, (1, 0, 2))
                Save_BIM(np.int32(Segmentation_Result), self.outputpath + self.subject_name + '_seg.BIM',
                         input_file=self.IM0path)
                saved_files.append(f"{self.subject_name}_seg.BIM")
            else:
                save(np.int32(Segmentation_Result), self.outputpath + self.subject_name + '_seg.nii.gz', hdr=self.header)
                saved_files.append(f"{self.subject_name}_seg.nii.gz")
            
            # -------------------save stl--------------------------------
            stl_saved = False
            try:
                savestl(Segmentation_Result, self.spacing, self.subject_name, self.outputpath)
                saved_files.append(f"{self.subject_name}.stl")
                stl_saved = True
            except Exception as e:
                print(f"STL save failed: {e}")
            
            # 显示汇总消息
            msg = f"Files saved to:\n{self.outputpath}\n\n"
            msg += "\n".join(saved_files)
            if not stl_saved:
                msg += "\n\nNote: STL export failed (surface level issue)"
            
            QMessageBox.information(self.QMainWindow, "Save Successful", msg)
                
        except Exception as e:
            print(f"Save failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.QMainWindow, "Save Failed", f"Error during save:\n{str(e)}")

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
        # 根据 label 设置颜色：label_0为绿色，label_1为红色
        if point[3] == 1:
            actor.GetProperty().SetColor(1, 0, 0)  # 红色
        else:
            actor.GetProperty().SetColor(0, 1, 0)  # 绿色
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
    setDirPath("./testdata/DCM_Data/01")

    app = QtWidgets.QApplication(sys.argv)
    Widget = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Widget)
    Widget.show()
    sys.exit(app.exec_())
