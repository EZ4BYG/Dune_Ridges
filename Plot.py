#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from collections import Counter
import matplotlib
import scipy


# In[5]:


img = glob.glob( r'D:\SGDownload\611_早上\未命名(2)\未命名(2)_大图\待处理png图\沙漠数据集4_100\处理后标签\*' )
img = sorted(img, key = lambda x:int(x.split('\\')[-1].split('.')[0].split('_')[-1]) )  # 排序
total = len(img)
save_path = 'D:/SGDownload/611_早上/未命名(2)/未命名(2)_大图/待处理png图/沙漠数据集4_100/标签换种显示方式/'

for x in range(total):
    tmp = tf.io.read_file(img[x])
    tmp = tf.image.decode_png(tmp, channels = 1)
    tmp = tmp.numpy().reshape(256,256)
    
    plt.imsave(save_path + 'subimg_{}.png'.format(x), tmp)


# In[4]:


plt.imshow( img )


# In[5]:


path = 'D:/SGDownload/611_早上/未命名(2)/未命名(2)_大图/待处理png图/沙漠数据集4_100/subimg_1.png'

plt.imsave(path, img)


# In[31]:


label = glob.glob( r'D:\SGDownload\611_早上\未命名(2)\未命名(2)_大图\待处理png图\沙漠数据集12_450\处理后标签\subimg_2371.png' )
img = glob.glob( r'D:\SGDownload\611_早上\未命名(2)\未命名(2)_大图\待处理png图\沙漠数据集12_450\图像\subimg_2371.png' )

label = tf.io.read_file(label[0])
label = tf.image.decode_png(label, channels = 1)
label = label.numpy().reshape(256,256)

img = tf.io.read_file(img[0])
img = tf.image.decode_png(img, channels = 3)
img = img.numpy().reshape(256,256,3)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(label)
plt.subplot(1,2,2)
plt.imshow(img)


# In[32]:


plt.imsave( 'D:/SGDownload/611_早上/未命名(2)/未命名(2)_大图/待处理png图/沙漠数据集12_450/subimg_2371.png',label )


# In[22]:


plt.imshow(label)


# In[ ]:




