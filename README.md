# lidar_instance_segmentation_tvm

## Getting Started
1) Build and install tvm  
Please refer to https://tvm.apache.org/docs/install/from_source.html  
Download pre-build bainaries llvm or build it from source.  https://releases.llvm.org/download.html 
and edit `build/config.cmake`
```
set(USE_CUDA ON)
set(USE_LLVM /path/to/your/llvm/bin/llvm-config)
```

2 )Clone and copy the following package
- [autoware_perception_msgs](https://github.com/tier4/autoware.iv.universe/tree/master/common/msgs/autoware_perception_msgs)  
- [dynamic_object_visualization](https://github.com/tier4/autoware.iv.universe/tree/master/perception/util/visualizer/dynamic_object_visualization)  

```
cd <your catkin_ws>/src
git clone https://github.com/tier4/lidar_instance_segmentation_tvm.git
git clone https://github.com/tier4/autoware.iv.universe.git /tmp/autoware.iv.universe
cp -r /tmp/autoware.iv.universe/common/msgs/autoware_perception_msgs .
cp -r /tmp/autoware.iv.universe/perception/util/visualizer/dynamic_object_visualization .
catkin b
```

3) Download the pre-trained model and compile the tvm model  
Requires python 3.7 or higher.
```
pip install -r requirements.txt
python prepare_model.py
```

4) Run
```
roslaunch lidar_instance_segmentation_tvm lidar_instance_segmentation.launch
```
