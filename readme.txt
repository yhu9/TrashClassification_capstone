Required libraries to use software.
---------------------------------------------
python 2.7.6
bash 4.3.11
python-opencv 3.0.0+
numpy
matplotlib
Tkinter 8.6
svm binaries (http://svmlight.joachims.org) x64

The software was last run on a machine running Ubuntu 16.04 on an x64 system. Make sure your downloads are for your computer architecture since the svm_light used in this project is for an x64 system and will not work for an x32 system. It is recommended that you run the software on a linux based system running bash. Also, make sure to download opencv from so

To install python:

sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install python2.7

To install python-opencv 3.0.0+ from source:

  http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

To install numpy:

  sudo apt-get install python-numpy

To install matplotlib:

  sudo apt-get install python-matplotlib

To install Tkinter 8.6:

  sudo apt-get install python python-tk idle python-pmw python-imaging

To install svm_binaries:

  http://svmlight.joachims.org



IMPORTANT:

- Make sure to download python-opencv from the source code found on the github repository as the apt-repo contains a much older version which does not have the features needed to run this software

- Make sure to download the correct svm_light binary for your computer architecture




