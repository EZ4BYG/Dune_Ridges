# 程序说明

整个工作分为3个部分：预处理部分、网络训练部分、预测部分

### 预处理部分：
- 首先下载研究区域17级精度三通道彩色大图（一张整图） —— Original_Dataset文件夹
- 使用Preprocess.ipynb程序对大图进行分割（分割成单位大小256x256x3）和子图重命名；
- 使用Random_Sampling.ipynb程序进行训练集的随机采样，即获取一小部分需进一步进行“人工加标签”的图像集；—— Train_Dataset文件夹
- 再次使用Preprocess.ipynb程序中的部分功能将训练集图像转为“二值图”，即图像只有标签和背景两个值；然后该程序中还包含对特殊图像的标签处理，若使用到可详细查看；
p.s. 数据预处理完成后，会得到Original_Dataset和Train_Dataset文件夹，后面训练部分只使用Train_Dataset文件夹中Preprocessed_Images中的图像数据。

### 网络训练部分：
- 使用Main_Unet.ipynb程序，以Preprocessed_Images文件夹中图像为训练集进行训练。该程序是网络训练的主函数，其中包含各种网络训练开始前的预处理工作，可自由调整（包括图像增强、网络结构、参数保存、训练过程的检查点等）；
p.s. 网络训练部分结束后，可保存得到一个Model.h5的训练好的网络参数模型；

### 预测部分：
- 程序Prediction.ipynb可以对所有其他子图进行标签预测；
- 程序Prediction_Interlayer.ipynb可以对其他子图在网络中的每个中间层输出进行可视化，即可了解网络每个中间层的预测结果；该程序也是文中关于“中间层”部分的绘图函数；
- 程序Plot.ipynb主要负责额外的“原图-预测结果”的拼接显示，非主要函数；
- 可再次使用Preprocess.ipynb程序中的“子图合并”功能将所有子图的预测结果进行合并，得到整个研究区域的完整预测结果图。
p.s. 预测部分结束后，可得到Prediction_Results文件夹，里面是所有子图的预测结果。

----

配置说明：
1. .ipynb是jupyter notebook保存的python程序，可直接在网页中一步一步运行并得到结果（推荐）。若不习惯这种方法，也可使用同名的.py原程序；
2. Tensorflow 2.0+, Anaconda 2020+
