# MedSAMate
Semi-Automatic Medical Image Segmentation Software Based on SAM and Hybrid Intelligence
# SAM-MeD 分割方法软件包使用说明

## 软件文件信息
| 文件名 | 类型 | 大小 |
| ---- | ---- | ---- | 
| SAMMed Viewer.exe | 应用程序 | 37,550 KB |

<img src="./images/图片1.png" width="450" height="300" alt="软件主界面">
<img src="./images/图片2.png" width="450" height="300" alt="功能概览">

## 操作步骤

### (a) 打开文件
1. 启动 SAM-Med Viewer 软件，在菜单栏中选择【File】→【Load IM0】；  
   <img src="./images/图片3.png" width="350" height="250" alt="打开IM0文件">

2. 选择需要处理的 “.IM0” 格式文件；  

3. 在软件界面的【Subject Name】输入框中，填写受试者名称；  
   <img src="./images/图片4.png" width="350" height="250" alt="填写受试者名称">

4. 可通过【Contrast Adjustment】（对比度调整）、【Window Level】（窗位）、【Window Width】（窗宽）功能优化图像显示效果。


### (b) 交互式标注工具
1. 在菜单栏中选择【SAM-Med2D Seg】；  
   <img src="./images/图片5.png" width="350" height="250" alt="选择分割功能">

2. 勾选【Bounding-box】（边界框）功能；  
   <img src="./images/图片6.png" width="350" height="250" alt="启用边界框工具">

3. 对每个切片（Slice），手动设置感兴趣区域（ROI）；  
   <img src="./images/图片7.png" width="350" height="250" alt="标注ROI区域">


### (c) 分割 IM0 文件
1. 完成所有切片的 ROI 标注后，点击菜单栏中的【Start Segmentation】（开始分割）；  
   <img src="./images/图片8.png" width="350" height="250" alt="启动分割">

2. 等待软件自动执行分割运算（分割过程中请保持软件处于运行状态）；  
   <img src="./images/图片9.png" width="350" height="250" alt="分割过程">


### (d) 保存分割结果（BIM 格式）
1. 分割完成后，在菜单栏中选择【Save Results as BIM】；  
   <img src="./images/图片10.png" width="350" height="250" alt="保存BIM文件">

2. 软件默认将分割结果保存至 “output” 文件夹中；  

3. 生成的 BIM 文件命名示例：VGC072_IS_In_seg.BIM（文件大小示例：247 KB），修改时间与分割完成时间一致；  
   <img src="./images/图片11.png" width="350" height="400" alt="BIM文件示例">


## 补充说明
- 支持的输入文件格式：.IM0、.DICOM、.STL（通过【Load DICOM】【Load STL】功能加载）；  
- 标注工具还支持【Point】（点标注）及【Point Label】（点标签）功能，可根据需求选择使用；
