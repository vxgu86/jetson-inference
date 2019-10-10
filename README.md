<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

主要用到了英伟达NVIDIA的 **[DIGITS](https://github.com/NVIDIA/DIGITS)** 和 **[Jetson Xavier/TX1/TX2/nano](http://www.nvidia.com/object/embedded-systems.html)** ，DIGITS用于以可视化交互的方式在云或工作站上的标注数据集上训练网络模型，Jetson Xavier/TX1/TX2运行推理（Inference），是嵌入式的推理部署端。

# 深度学习的部署

NVIDIA的GPU是训练深度学习模型的不二之选，且已形成了庞大的生态环境。主要针对NVIDIA嵌入式平台 **[Jetson Nano/TX1/TX2/Xavier](http://www.nvidia.com/object/embedded-systems.html)的推理和实时[DNN 视觉](#api-reference) 库**。 用 **[TensorRT](https://developer.nvidia.com/tensorrt)** 将神经网络高效地部署到嵌入式平台上，利用图优化（graph optimizations），内核融合（kernel fusion）和FP16/INT8等方式提高性能和功率效率。

视觉方面的应用基础模块，主要有用于图像识别的[`imageNet`](c/imageNet.h)，用于对象定位的[`detectNet`](c/detectNet.h)和用于语义分割的[`segNet`](c/segNet.h)，都是从[`tensorNet`](c/tensorNet.h)对象继承的，同时提供了对输入图像和摄像机实时视频流进行处理的示例实现，C++ 和 Python库的参考文档具体参见 **[API Reference](#api-reference)**。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-primitives.png" width="800">

这本书的使用上可以按照需求来学习不同的模块，既包括用DIGITS进行模型的训练，也有Jetson平台上开展的推理和板上迁移学习（transfer learning）。

### Table of Contents

* [内容目录](#内容目录)
* [API Reference](#api-reference)
* [代码示例](#代码示例)
* [预训练模型](#预训练模型)
* [系统要求](#推荐系统要求)
* [其他资源](#其他资源)

> &gt; &nbsp; Jetson Nano 开发套件和JetPack 4.2.2均支持 <br/>
> &gt; &nbsp; 参见技术blog，包括基准测试benchmarks, [`Jetson Nano Brings AI Computing to Everyone`](https://devblogs.nvidia.com/jetson-nano-ai-computing/). <br/>
> &gt; &nbsp; 板上已支持Python 和PyTorch片上训练!

## 内容目录

包括在云端或本地训练模型，在Jetson上用TensorRT进行推理，包括用TensorRT推理和用PyTorch进行迁移学习。可以用C++ 或者 Python编写自己的图像分类、目标检测等应用，带有实时摄像头的案例。

* [DIGITS 工作流程](docs/digits-workflow.md) 
* DIGITS 系统配置
	* [配置系统及在docker中安装DIGITS](docs/digits-setup.md)
	* [配置系统及在本机安装DIGITS](docs/digits-native.md)
* 用JetPack刷Jetson
	* [JetPack-L4T刷机](docs/jetpack-setup.md)
	* [NVIDIA SDK Manager刷机](docs/jetpack-setup-2.md)
* [编译项目源码](docs/building-repo.md)
* [用ImageNet进行图像分类](docs/imagenet-console-2.md)
	* [用C++编写自定义的图像分类应用](docs/imagenet-example-2.md)
	* [用Python编写自定义的图像分类应用](docs/imagenet-example-python-2.md)
	* [在实时摄像头输入上进行图像识别](docs/imagenet-camera.md)
	* [用DIGITS重训练网络](docs/imagenet-training.md)
	* [自定义目标类别](docs/imagenet-training.md#customizing-the-object-classes)
	* [在Jetson上部署模型快照](docs/imagenet-snapshot.md)
	* [在Jetson上加载自定义模型](docs/imagenet-custom.md)
* [用DetectNet进行物体检测定位](docs/detectnet-training.md)
* [用DetectNet进行物体检测定位](docs/detectnet-console-2.md)
	* [在DIGITS中训练目标检测模型](docs/detectnet-training.md)
	* [将模型部署到Jetson及其修改](docs/detectnet-snapshot.md)
	* [（多）目标检测模型运行](docs/detectnet-console.md)
	* [在实时摄像头输入上进行物体检测](docs/detectnet-camera-2.md)
	* [编写自定义的物体检测应用](docs/detectnet-example-2.md)
* [用SegNet进行语义分割](docs/segnet-dataset.md)
* [用SegNet进行语义分割](docs/segnet-console-2.md)
	* [Aerial Drone Dataset数据集处理](docs/segnet-dataset.md)
	* [Generating Pretrained FCN-Alexnet](docs/segnet-pretrained.md)
	* [在DIGITS中训练FCN-Alexnet](docs/segnet-training.md)
	* [FCN-Alexnet Patches for TensorRT](docs/segnet-patches.md)
	* [在Jetson上运行模型](docs/segnet-console.md)
	* [在实时摄像头输入上进行语义分割](docs/segnet-camera-2.md)	
* [用PyTorch进行迁移学习](docs/pytorch-transfer-learning.md)
	* [在Cat/Dog数据集上进行重训练](docs/pytorch-cat-dog.md)
	* [在PlantCLEF数据集上进行重训练](docs/pytorch-plants.md)
	* [构建自己的数据集](docs/pytorch-collect.md)
	
## API Reference

[C++](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/index.html) 和 [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.html)库的参考文档。

#### jetson-inference

|                   | [C++](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/group__deepVision.html) | [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html) |
|-------------------|--------------|--------------|
| Image Recognition | [`imageNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/classimageNet.html) | [`imageNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#imageNet) |
| Object Detection  | [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/classdetectNet.html) | [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet)
| Segmentation      | [`segNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/classsegNet.html) | [`segNet`](https://rawgit.com/dusty-nv/jetson-inference/pytorch/docs/html/python/jetson.inference.html#segNet) |

#### jetson-utils

* [C++](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/group__util.html)
* [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.utils.html)

在外部工程中可以通过链接`libjetson-inference` and `libjetson-utils`来使用这些库。

## 代码示例

代码库的使用示例可见：

* [Coding Your Own Image Recognition Program (Python)](docs/imagenet-example-python-2.md)
* [Coding Your Own Image Recognition Program (C++)](docs/imagenet-example-2.md)

更多C++ and Python的示例可见： 

|                   | Images              | Camera              |
|-------------------|---------------------|---------------------|
| **C++ ([`examples`](examples/))** | |
| &nbsp;&nbsp;&nbsp;Image Recognition | [`imagenet-console`](examples/imagenet-console/imagenet-console.cpp) | [`imagenet-camera`](examples/imagenet-camera/imagenet-camera.cpp) |
| &nbsp;&nbsp;&nbsp;Object Detection  | [`detectnet-console`](examples/detectnet-console/detectnet-console.cpp) | [`detectnet-camera`](examples/detectnet-camera/detectnet-camera.cpp)
| &nbsp;&nbsp;&nbsp;Segmentation      | [`segnet-console`](examples/segnet-console/segnet-console.cpp) | [`segnet-camera`](examples/segnet-camera/segnet-camera.cpp) |
| **Python ([`python/examples`](python/examples/))** | | |
| &nbsp;&nbsp;&nbsp;Image Recognition | [`imagenet-console.py`](python/examples/imagenet-console.py) | [`imagenet-camera.py`](python/examples/imagenet-camera.py) |
| &nbsp;&nbsp;&nbsp;Object Detection  | [`detectnet-console.py`](python/examples/detectnet-console.py) | [`detectnet-camera.py`](python/examples/detectnet-camera.py) |
| &nbsp;&nbsp;&nbsp;Segmentation      | [`segnet-console.py`](python/examples/segnet-console.py) | [`segnet-camera.py`](python/examples/segnet-camera.py) |

> **注意**:  涉及到使用numpy arrays, 可参考 [`cuda-from-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-numpy.py) 和 [`cuda-to-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-numpy.py)

这些示例在These examples will automatically be compiled while [编译项目源码](docs/building-repo-2.md)时会被自动编译, 可以直接运行下面列出的预训练模型以及我们自定义的模型，在运行时输入`--help`可以输出使用说明信息。

## 预训练模型

包含很多预训练模型，可以通过[**Model Downloader**](docs/building-repo-2.md#downloading-models) 下载。

#### 图像识别

|   网络        | CLI argument   | NetworkType enum |
| --------------|----------------|------------------|
| AlexNet       | `alexnet`      | `ALEXNET`        |
| GoogleNet     | `googlenet`    | `GOOGLENET`      |
| GoogleNet-12  | `googlenet-12` | `GOOGLENET_12`   |
| ResNet-18     | `resnet-18`    | `RESNET_18`      |
| ResNet-50     | `resnet-50`    | `RESNET_50`      |
| ResNet-101    | `resnet-101`   | `RESNET_101`     |
| ResNet-152    | `resnet-152`   | `RESNET_152`     |
| VGG-16        | `vgg-16`       | `VGG-16`         |
| VGG-19        | `vgg-19`       | `VGG-19`         |
| Inception-v4  | `inception-v4` | `INCEPTION_V4`   |

#### 目标检测

| 网络                    | CLI argument       | 网络类型枚举        | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 ([COCO classes](data/networks/ssd_coco_labels.txt)) |
| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 ([COCO classes](data/networks/ssd_coco_labels.txt)) |
| SSD-Inception-v2        | `ssd-inception-v2` | `SSD_INCEPTION_V2` | 91 ([COCO classes](data/networks/ssd_coco_labels.txt)) |
| DetectNet-COCO-Dog      | `coco-dog`         | `COCO_DOG`         | dogs                 |
| DetectNet-COCO-Bottle   | `coco-bottle`      | `COCO_BOTTLE`      | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`       | `COCO_CHAIR`       | chairs               |
| DetectNet-COCO-Airplane | `coco-airplane`    | `COCO_AIRPLANE`    | airplanes            |
| ped-100                 | `pednet`           | `PEDNET`           | pedestrians          |
| multiped-500            | `multiped`         | `PEDNET_MULTI`     | pedestrians, luggage |
| facenet-120             | `facenet`          | `FACENET`          | faces                |

#### 语义分割

| 数据集      | 分辨率 | CLI Argument | Accuracy | Jetson Nano | Jetson Xavier |
|:------------:|:----------:|--------------|:--------:|:-----------:|:-------------:|
| [Cityscapes](https://www.cityscapes-dataset.com/) | 512x256 | `fcn-resnet18-cityscapes-512x256` | 83.3% | 48 FPS | 480 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 1024x512 | `fcn-resnet18-cityscapes-1024x512` | 87.3% | 12 FPS | 175 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 2048x1024 | `fcn-resnet18-cityscapes-2048x1024` | 89.6% | 3 FPS | 47 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 576x320 | `fcn-resnet18-deepscene-576x320` | 96.4% | 26 FPS | 360 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 864x480 | `fcn-resnet18-deepscene-864x480` | 96.9% | 14 FPS | 190 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 512x320 | `fcn-resnet18-mhp-512x320` | 86.5% | 34 FPS | 370 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 640x360 | `fcn-resnet18-mhp-512x320` | 87.1% | 23 FPS | 325 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 320x320 | `fcn-resnet18-voc-320x320` | 85.9% | 45 FPS | 508 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 512x320 | `fcn-resnet18-voc-512x320` | 88.5% | 34 FPS | 375 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 512x400 | `fcn-resnet18-sun-512x400` | 64.3% | 28 FPS | 340 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 640x512 | `fcn-resnet18-sun-640x512` | 65.1% | 17 FPS | 224 FPS |

* 如果CLI argument没有输入分辨率, 默认使用最低分辨率。
* 精度Accuracy表示模型验证数据集上的像素分类精度。
* 性能是在GPU FP16 mode with JetPack 4.2.1, `nvpmodel 0` (MAX-N)模式下测试得到的。

<details>
<summary>Legacy Segmentation Models</summary>

| Network                 | CLI Argument                    | NetworkType enum                | Classes |
| ------------------------|---------------------------------|---------------------------------|---------|
| Cityscapes (2048x2048)  | `fcn-alexnet-cityscapes-hd`     | `FCN_ALEXNET_CITYSCAPES_HD`     |    21   |
| Cityscapes (1024x1024)  | `fcn-alexnet-cityscapes-sd`     | `FCN_ALEXNET_CITYSCAPES_SD`     |    21   |
| Pascal VOC (500x356)    | `fcn-alexnet-pascal-voc`        | `FCN_ALEXNET_PASCAL_VOC`        |    21   |
| Synthia (CVPR16)        | `fcn-alexnet-synthia-cvpr`      | `FCN_ALEXNET_SYNTHIA_CVPR`      |    14   |
| Synthia (Summer-HD)     | `fcn-alexnet-synthia-summer-hd` | `FCN_ALEXNET_SYNTHIA_SUMMER_HD` |    14   |
| Synthia (Summer-SD)     | `fcn-alexnet-synthia-summer-sd` | `FCN_ALEXNET_SYNTHIA_SUMMER_SD` |    14   |
| Aerial-FPV (1280x720)   | `fcn-alexnet-aerial-fpv-720p`   | `FCN_ALEXNET_AERIAL_FPV_720p`   |     2   |

</details>

## 推荐系统要求

**训练时所需GPU:**  Maxwell, Pascal, Volta, 或者 Turing体系的GPU (理想情况下最少6GB video memory)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可以选择AWS P2/P3 instance 或者 Microsoft Azure N-series  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 16.04/18.04 x86_64

**部署:**    &nbsp;&nbsp;Jetson Nano Developer Kit with JetPack 4.2 or newer (Ubuntu 18.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson Xavier Developer Kit with JetPack 4.0 or newer (Ubuntu 18.04 aarch64)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX2 Developer Kit with JetPack 3.0 or newer (Ubuntu 16.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX1 Developer Kit with JetPack 2.3 or newer (Ubuntu 16.04 aarch64).

这里的tensorRT示例是为部署在Jetson上的，但若通用计算机上安装了cuDNN and TensorRT，这些tensorRT示例可以编译在计算机上。

## 其他资源

* [ros_deep_learning](http://www.github.com/dusty-nv/ros_deep_learning) - TensorRT inference ROS nodes
* [NVIDIA AI IoT](https://github.com/NVIDIA-AI-IOT) - NVIDIA Jetson GitHub repositories
* [Jetson eLinux Wiki](https://www.eLinux.org/Jetson) - Jetson eLinux Wiki

##
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="#deploying-deep-learning"><sup>Table of Contents</sup></a></p>


术语：
deep neural networks (DNNs)

训练 training

推理（Inference）

classification

detection

语义分割 Semantic Segmentation

transfer learning
