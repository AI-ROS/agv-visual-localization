# Produtech
Core packages for the Produtech Project

# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
  * [1: Hardware](#hardware) 
    * [Robot](#robot)
    * [Cameras](#cameras)
    * [Jetson AGX Xavier](#computer)
  * [2: Software](#software)
    * [maxon_des](#maxon-des)
    * [ros-maxon-driver](#ros-maxon-driver)
    * [ros-panorama-package](#ros-panorama)
    * [faster-rcnn-data-matrix](#data-matrix-detection)
    * [deep-stream-application](#deepstream-app)
- [Known problems](#known-problems)
  * [High latency on the panorama image creation](#panorama-problem)
- [Publications](#publications)
- [Future Work](#future-work)

# Overview

One of the branches of the Produtech II SIF 24541 project is the T.6.3.3 - Development of a flexible and low-cost localization and navigation system for PPS6. So, we aim to develop the core features of a visual-based navigation system. 
The system is composed of a small robot controlled through a remote controller) that emulates an industrial AGV and a set of programs. This robotic system would be able to detect landmarks (encoded Data Matrix), which are spread in the environment trying to create a constellation in such a way that several of them are always visible to a set of cameras onboard the robot or AGV. These markers are encoded with their location relative to a known reference. Then, the robot localization can be computed by applying triangulation and trilateration techniques. 


# Setup 

The setup of our robot is composed of two main parts: the hardware and the software. Regarding the hardware, we retrofitted the Atlas MV robot by disassembling the old robot and leaving only the interesting parts for the current project. Also, we renewed the existing software in terms of communication between the power chart and the engine. Finally, we built some of the novel core programming modules to perform the real-time self-robot localization.  

## 1: Hardware

Here, we describe the hardware parts and software modules used and developed during this project.

### Robot

In terms of hardware retrofitting we changed the entire old electronic by a simpler and actual one (e.g. the usage of an Arduino to do the communication between the Joystick and the steering AC motor). One of the initial setups of the robot can be seen in the figure below:

<p align="center">
  <img width="620" height="543" src="docs/robot.png">
</p>



We have not an image of the final setup but we describe some of the new hardware parts that compose the final robot version. 


Name  | Description/Function
:---: | :---:
DC/AC Inversor | Input 48 V, Output: AC. To power the Jetson.
DC/DC Inversor | Input 12 V, Output: 5V. To power the arduino.
Arduino |  To perform the communication between the remote controller and the AC motor.
Jetson AGX Xavier  | To perform the DL computation and to ensure the ROS architecture running.
Four Cameras | To acquire data.

### Cameras

The cameras used to acquire the images can be seen in the figure below:

<p align="center">
  <img width="257" height="377" src="docs/cameras_xavier.png">
</p>

The [e-CAM130_CUXVR - Multiple Camera Board](https://www.e-consystems.com/nvidia-cameras/jetson-agx-xavier-cameras/four-synchronized-4k-cameras.asp) were conceived to acquire images with the [NVIDIA® Jetson AGX Xavier™](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit) board.


### Jetson AGX Xavier

This board enables the creation of AI applications mainly based on Deep Learning by incorporating 512-core Volta GPU with Tensor Cores and (2x) NVDLA Engines. On this board is installed the Nvidia [Jetpack 4.2](https://developer.nvidia.com/jetpack-4_2) and the [DeepStream SDK 4.0](https://docs.nvidia.com/metropolis/deepstream/4.0/dev-guide/DeepStream_Development_Guide/baggage/index.html). 