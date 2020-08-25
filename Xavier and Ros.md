# Xavier NX 常用工具
**系统烧录**   
https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#prepare 

**动态检测内存**   
```
sudo -H pip install jetson-stats   
sudo jtop   
```
**功率设置**   
```
Power Mode
修改Power Mode状态
$ sudo nvpmodel -m <x>   #Where <x> is 0/1/... 2 is max
修改风扇运行模式
$ sudo /usr/sbin/nvpmodel -d <fan_mode>   #Where <fan_mode> is quiet or cool.
查看当前状态
$ sudo nvpmodel -q
```
功率分档见官网
https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html   

**配置pytorch**    
```
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython numpy
pip3 install torch-1.6.0.whl
torch 1.6版本还没有对应版本的torchvison，直接https://github.com/pytorch/vision 从源码编译
```

# ROS系统

### Install

**(1)官网安装-只支持Python2**
http://wiki.ros.org/cn/melodic/Installation/Ubuntu
**(2)安装python3需要从源码编译**
https://zhuanlan.zhihu.com/p/77682229
注意: 需要修改源码中/src/catkin/bin下面的文件的第一行环境为python3  #!/usr/bin/python3

### 创建与编译工程
```
（1）cd catkin_ws/src  # 进入工作空间下的 src 目录  
（2）catkin_create_pkg <package_name> [depend1] [depend2] [depend3]
（3）修改package.xml CMakeLists.txt中的依赖项
（4）cd ..             #退到 catkin_ws 工作空间路径下    
（5）catkin_make       #编译软件包  
（6）roscd <package_name> && mkdir scripts && cd scripts (放入代码文件)
（7）chmod +x talker.py listener.py #必须先加执行权限
（8）roscore
（9）rosrun <package_name> talker.py

echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
### 消息结构体构造
rosmsg list #列出支持的所有类型
rosmsg show std_msgs/Int8   #可以show出消息的构造


# 常见报错与处理方法

**（1）如何传递一个数组**
可以使用float32multiarray.data,也可以自己定义个消息，数据类型 float32[] 就可以传递数组了，但是必须是一维的。如果传递多维数组，需要先拉平成一维 flatten()

**（2）如何传输图像**
通过sensor_msgs.msg 中的 Image，CompressedImage
http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber

**（3）消息中包含多个内容—构造结构体msg**
http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_msg
http://wiki.ros.org/ROS/Tutorials/CustomMessagePublisherSubscriber%28python%29

**（4）如何使用回调函数获取的数据**
可以将一些需要的变量声明为全局变量，或者用类的方式写callback从而所有实例变量都可以调用
https://stackoverflow.com/questions/37373211/update-the-global-variable-in-rospy

**（5）调用usb_cam node**
git clone https://github.com/ros-drivers/usb_cam.git && cd~/catkin_ws && catkin_make
roslaunch usb_cam usb_cam_test.launch
launch文件中可以调整图像发布频率 产生的消息名/usb_cam/image_raw，可以使用image_transport库压缩图像后再传输

**（6）opencv图像与ROS图像消息之间的转化**
通过cv_bridge
http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
编译遇到的问题是仅支持OPENCV3 修改为OPENCV4版本
I was able to successfully compile cv_bridge with opencv4:
step1: modify src/vision_opencv/cv_bridge/CMakeLists.txt 3->4
find_package(OpenCV 3 REQUIRED
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
 CONFIG
step2: Add set (CMAKE_CXX_STANDARD 11) to your top level cmake
step3: In cv_bridge/src CMakeLists.txt line 35 change to if (OpenCV_VERSION_MAJOR VERSION_EQUAL 4)
step4: In cv_bridge/src/module_opencv3.cpp change signature of two functions
```
4.1) UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const
->to->
UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const
4.2) bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
->to->
bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const
```
编译
```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so
```

**(7)ROS Image subscriber lag 延时问题（收发之间产生的滞后）**
第一点是要设置Publisher的queue_size等于1；
第二点是要设置Subscriber的queue_size（消息队列大小）等于1；
第三点非常重要，要设置Subscriber的buff_size（缓冲区大小）足够大，大于一个消息的大小。
rospy.Subscriber('det_msg', Detection, callback, queue_size=1, buff_size=2**24)

##常见报错
1.install pillow
```
->error
The headers or library files could not be found for jpeg.
->solution
sudo apt-get install libjpeg-dev
```
2.install pycuda
```
->error
nvcc not found
->solution
添加环境变量， vim ~/.bashrc ，添加环境变量export PATH=$PATH:/usr/local/cuda/bin
```

3.build alphapose
```
->error
command 'aarch64-linux-gnu-gcc' failed with exit status 1
alphapose/utils/roi_align/src/roi_align_cuda.cpp:20:23: error: ‘AT_CHECK’ was not declared in this scope
 #define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
->solution
替换alphapose中AT_CHECK 为TORCH_CHECK
(pytorch 1.5 之后已经抛弃AT_CHECK，换用TORCH_CHECK）
```

4.pip install
```
->error
ImportError: No module named 'pip._internal'
->solution
updating pip via Python, like this:
python3 -m pip install --user --upgrade pip
```

5.yaml
```
->error
module 'yaml' has no attribute 'FullLoader'
->solution
The FullLoader class is only available in PyYAML 5.1 and later. Version 5.1 
pip install -U PyYAML
->error
Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
->solution
pip 10 no longer uninstalls distutils packages. So I downgraded to pip 8.1.1. And now it works.
sudo -H pip3 install pip==8.1.1
```
6.
```
ModuleNotFoundError: No module named 'beginner_tutorials'
source devel/setup.bash
```
7.sensor_msgs
```
The error message indicates that you use PoseArray without prefixing it with geometry_msgs.
Just to clarify the syntax in a message file - it is:
PackageName/VariableType variable_name
The PackageName/ is left out when referring to message from the same package.
```
8.
```
import-im6.q16: not authorized `rospy' @ error/constitute.c/WriteImage/1037.
from: can't read /var/mail/std_msgs.msg
from: can't read /var/mail/sensor_msgs.msg
from: can't read /var/mail/yolo_sppe.msg
import-im6.q16: not authorized `os' @ error/constitute.c/WriteImage/1037.
import-im6.q16: not authorized `time' @ error/constitute.c/WriteImage/1037.
import-im6.q16: not authorized `argparse' @ error/constitute.c/WriteImage/1037.
->solution
'must add #!/usr/bin/env python3  in the top of code.
```
9.torch2onnx
```
nx@nx-desktop:~/Documents/tensorrt_demos/yolov3_onnx$ python3 yolov3_to_onnx.py --model yolov3-288
Parsing DarkNet cfg file...
Building ONNX graph...
Traceback (most recent call last):
  File "yolov3_to_onnx.py", line 869, in <module>
    main()
  File "yolov3_to_onnx.py", line 854, in main
    verbose=True)
  File "yolov3_to_onnx.py", line 454, in build_onnx_graph
    params)
  File "yolov3_to_onnx.py", line 287, in load_upsample_scales
    name, TensorProto.FLOAT, shape, data)
  File "/usr/local/lib/python3.6/dist-packages/onnx/helper.py", line 173, in make_tensor
    getattr(tensor, field).extend(vals)
TypeError: 1.0 has type numpy.float32, but expected one of: int, long, float
->solution
using protobuf-3.8.0
```
10.onnx->tensorrt
```
 trtexec --onnx=simple_mobv3.onnx --minShapes=input:1x3x256x192 --optShapes=input:6x3x256x192 --maxShapes=input:8x3x256x192 --explicitBatch --saveEngine=model.engine
 trtexec --loadEngine=model.engine --optShapes=input:8x3x256x192
```

