# Produtech
Core packages for the Produtech Project

# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Materials](#setup)
  * [1: Hardware](#mat-hardware) 
    * [Robot](#atlas-mv)
    * [Cameras](#atlas-mv)
    * [Jetson AGX Xavier](#computer)
  * [2: Software](#mat-software)
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
The system is composed of a small robot that emulates an industrial AGV and a set of programs. This robotic system would be able to detect landmarks (encoded Data Matrix), which are spread in the environment trying to create a constellation in such a way that several of them are always visible to a set of cameras onboard the robot or AGV. These markers are encoded with their location relative to a known reference. Then, the robot localization can be computed by applying triangulation and trilateration techniques. 


# Setup 

The setup of our robot is composed of two main parts: the hardware and the software. Regarding the hardware, we retrofitted the Atlas MV robot by disassembling the old robot and leaving only the interesting parts for the current project. Also, we renewed the existing software in terms of communication between the power chart and the engine. Finally, we built some of the novel core programming modules to perform the real-time self-robot localization.  




