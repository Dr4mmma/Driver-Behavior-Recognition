# Driver-Behavior-Recognition

Using Python Deep Learning algorithms in order to monitor drivers while on the road.

Based on Image Recognition.





<p align="center" style="margin-top=100px">
  <img src="https://i.ibb.co/tsXX68r/64391-Converted.png">
</p>



# Credits 

| Project | License  | 
| :---:   | :-: | 
| [VTuber_Unity](https://github.com/kwea123/VTuber_Unity/tree/3becbfa73d424d565d2750b22139ea373381fc3b) | [License](https://github.com/kwea123/VTuber_Unity/blob/3becbfa73d424d565d2750b22139ea373381fc3b/LICENSE) |
| [DBSE-monitor](https://github.com/altaga/DBSE-monitor) | [License](https://github.com/altaga/DBSE-monitor/blob/master/LICENSE) |
| [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)| [License](https://github.com/yinguobing/head-pose-estimation/blob/master/LICENSE)|
| [face-alignment](https://github.com/1adrianb/face-alignment)| [License](https://github.com/1adrianb/face-alignment/blob/master/LICENSE)|
| [GazeTracking](https://github.com/antoinelame/GazeTracking)| [License](https://github.com/antoinelame/GazeTracking/blob/master/LICENSE)|


# Installation

## Hardware
* OS: Ubuntu 16.04 (at least) or Windows 10 (Maybe MacOS as well).
* Runs on both CPU and GPU (at least CUDA 9.0)

## Software
* Download and Install 64-bit Anaconda 3
  * While installing Anaconda make sure that you check both options:
    Add Anaconda to my PATH environment variable
    Register Anaconda as my default Python
    Add Anaconda to System PATH and make it default Python
* Create Virtual Environment
  * Open the command prompt and execute the following command:
  
  `conda create --name opencv-env python=3.6`
* Activate the environment and installing packages
  * Activate virtual environment (See how the (opencv-env) appears before the prompt after this command): 
  
    `activate opencv-env`
  * Install OpenCV and other important packages continuing from the above prompt, execute the following commands:
  
    `pip install numpy scipy matplotlib scikit-learn jupyter`
    
    `pip install opencv-contrib-python`
    
    `python -m pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf`
    
* Test your installation
  * Open the python prompt on the command line by typing python on the command prompt and then execute the following:
  
    `import cv2`
    
    `cv2.__version__`
    
     `import dlib`
     
     `dlib.__version__`
* Other packages may be needed to get installed, but you're almost there!

# How to use it (inluding brief summary of what you see):

## 1. Running the project:
* Execute the following command to run the project:

  `python DBR.py --debug --cpu`
  
  or (if running on GPU):
  
  `python DBR.py --debug`
  
## 2. Face detection test:
By merging the following modules:
  * [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)
  * [face-alignment](https://github.com/1adrianb/face-alignment)
  * [GazeTracking](https://github.com/antoinelame/GazeTracking)
  
Which was already done in the following project: [VTuber_Unity](https://github.com/kwea123/VTuber_Unity), you will recognize the following appearing on your screen:

![picture alt](https://github.com/kwea123/VTuber_Unity/blob/master/images/debug_cpu.gif "CPU Model")
![picture alt](https://github.com/kwea123/VTuber_Unity/blob/master/images/debug_gpu.gif "GPU Model")

###### Gifs are taken by: [VTuber_Unity](https://github.com/kwea123/VTuber_Unity)

## 3. Determining where the driver's is looking (Right mirror, left mirror, Front mirror (road), Rear mirror, Dashboard, Center Console):
This was done by saving data of driver's head and gaze direction in order to use a classifier (KNN Classifier) to determine where the driver is looking.

<p align="center">
  <img src="https://im3.ezgif.com/tmp/ezgif-3-f32045e0fd6b.gif">
</p>

## 4. Using sequence classification methods in order to determine drivers mode just by looking at the mirrors (using mirrors in a normal way):
This was done by saving data of the previous method in order to use a classifier (KNN Classifier (sequence classification)) to determine whether the driver is driving normally or not.

<p align="center">
  <img src="https://im3.ezgif.com/tmp/ezgif-3-92d4b0b07992.gif">
</p>

## 5. Running drowsiness and distraction detector module:
This module was merged with the main project to understand driver's state (distracted, sleepy or normal), this module is already implemented as [DBSE-Monitor](https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness)

<p align="center">
  <img src="https://im3.ezgif.com/tmp/ezgif-3-61afd389c24d.gif">
</p>

###### Gifs are taken by: [DBSE-Monitor](https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness)

## 6. Defining if the driver is in a normal driving mode or not:
This was done by an if statement:
* Driver is looking at the mirrors probably, not sleepy and not distracted, then the driver is normal.
* Driver is not looking at the mirrors probably or sleepy or distracted, then the driver is apnormal.
Then if the driver is not normal for over 52 frames, an alarm goes out for 8 seconds to warn the driver to focus on the road.

## 7. Implementation:
This project will be ran on Jetson nano/xavier, as it is a mini computer that was made for AI projects.

<p align="center">
  <img src="https://elinux.org/images/9/93/Jetson-Xavier-NX-DevKit-Module.jpg">
</p>

###### Gifs are taken by: [elinux](https://elinux.org/Jetson_Xavier_NX)

# Final thoughts
I would consider this project finished, but there will be minor modifications to data gathered in order to assure high accuracy.
The final product will be available soon, and hopefully I can get a commercial product available at very cheap price.
I would love the opportunity to have a talk with a production company that can help me to get to a commercial product.

Thank you for your time, this code is available for any developer that wants to go further with this project.
Contact me for any further details at: mohammad.tayseer.comn@gmail.com

# Licenses
  * [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)
  * [face-alignment](https://github.com/1adrianb/face-alignment)
  * [GazeTracking](https://github.com/antoinelame/GazeTracking)
  * [DBSE-Monitor](https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness)
  * [VTuber_Unity](https://github.com/kwea123/VTuber_Unity)
