# Driver-Behavior-Recognition

Using Python Deep Learning algorithms in order to monitor drivers while on the road.




# Credits 

| Project | License  | 
| :---:   | :-: | 
| [VTuber_Unity](https://github.com/kwea123/VTuber_Unity/tree/3becbfa73d424d565d2750b22139ea373381fc3b) | [License](https://github.com/kwea123/VTuber_Unity/tree/3becbfa73d424d565d2750b22139ea373381fc3b/licenses) |
| [DBSE-monitor](https://github.com/altaga/DBSE-monitor) | [License](https://github.com/altaga/DBSE-monitor/blob/master/LICENSE) |

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
