# 图片全景拼接

代码来自[pyimagesearch](https://pyimagesearch.com)

# 原理简述

对多张图片进行基于`SIFT`的特征检测算法，如果符合最小拼接要求大的关键点`matchKeypoints`数量，使用`OpenCV-Python`自带的`stitching`方法进行全景拼接，但是对于拼接后的黑边裁剪效果不好，可以修改优化。

# 使用方式

`python image_stitching.py --images images/scottsdale --output output.png --crop 1`

其中`images/scottsdale`为待拼接图像所在文件夹，`output.png`为处理拼接保存后的图像；这里使用了相对路径，因为在工程根目录下运行了终端。不确定在根目录最好使用完整的绝对路径。` --crop 1`为是否裁剪黑色边框，缺省则不裁剪。

# 对比效果

相比其他方式，因为使用了较好的匹配关键点，所以拼接没有`裂缝`，没有`鬼影`

示例1：有裂缝（两张图片拼接中间有竖着的裂缝）

![](/img/20190902001913.png)

示例2：有鬼影（重影，由于拼接范围内有运动物体）

![](/img/20190902002008.png)

示例3：使用上面pyimagesearch的代码，没有裂缝和鬼影

![](/img/20190902002033.png)