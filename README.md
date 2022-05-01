# End-to-End Learning using CNN,CNN+LSTM, CNN+GRU, CNN+CT-GRU, CNN+NCP 

This repo provides the code of end-to-end learning in autonomous driving using different network structures. It includes network structure implementation, data collection, data processing, network training, test of trained policies , visualization of results.


## **Getting Started**
### Requirements
- python 3.8 
- the following dependencies (see below for an instruction to install them)
    - torch
    - torchvision
    - matplotlib==3.4.2
    - numpy==1.21.2
    - pandas==1.3.0
    - seaborn==0.11.1
    - h5py==3.3.0
    - pillow==8.3.1
    - opencv-python

### First steps (Ubuntu)

Clone the repository

`git clone https://gitlab.lrz.de/braindrive/bio_inspired_end_to_end.git`

Create the virtual environment and install all dependencies

`cd bio_inspired_end_to_end`  
`source setup.sh`


## **Network Structures**

In the folder `nets`, script `cnn_head.py` implements CNN used in the combination of CNN+RNN; script `models_all.py` implements CNN, LSTM, GRU, CT-GRU, NCP; script `ltc_cell.py` implements the LTC neuron used in NCP.
In the project folder, script `wiring.py` defines how the LTC neurons are connected in NCP. 

## **Data Collection**
Two datasets are used in this project:
One dataset is [CARLA driving data](http://rpg.ifi.uzh.ch/RAMNet.html), which is provided by Gehrig et al.
The other dataset is LGSVL driving data, collected by oneself in simulator LGSVL, using ros2 node to subscribe the images and the actions, script `ros_collect.py` is used in the ros package, to save LGSVL driving data in local folders.

## **Data processing**

In order to accelerate the training by improving the I/O speed of images, the collected data needs to be converted into HDF5 format. 
Script `hdf5_format_carla_data.py` is used to convert CARLA driving data into HDF5 format and script `hdf5_format_lgsvl_data.py` is used to convert LGSVL driving data into HDF5 format.     
To convert the format, run    
`python hdf5_format_carla_data.py --type argument` or `python hdf5_format_lgsvl_data.py --type argument`  
possible options of argument: train, valid, test for converting CARLA data; train, valid for converting LGSVL data.  
__Notice__: every time when converting data version, ensure there is no HDF5 file with the same name in the target folder.  

## **Network Training**
### Compare different convolutional head 
The comparison is peformed by training CNN+NCP in predicting steering, throttle, brake using CARLA driving data, Three CNN head are compared, they are NvidiaCNN, AlexNet, ResNet, implemented in `cnn_head.py`.  
Train networks by executing train_for_cnn_comp.py, run  
`python train_for_cnn_comp.py --cluster xxx --name xxx --epoch xxx --batch xxx --sequence xxx --hdf5 xxx --cnn_head xxx --seed xxx`  
Explanation of the arguments: cluster --> if using gpu or not; name --> file name to save results; epoch --> desired training epochs; batch --> the batch size; sequence --> time sequence length for NCP; hdf5 --> read data in original format or in hdf5; cnn_head --> the used type of CNN head; seed --> random seed for neural network initialization.

### Train CNN, CNN+LSTM, CNN+GRU, CNN+CT-GRU, CNN+NCP
Using CARLA driving data and LGSVL driving data to train CNN, CNN+LSTM, CNN+GRU, CNN+CT-GRU, CNN+NCP
run  
`python train_all_models.py --cluster xxx --data xxx --network xxx --name xxx --epoch xxx --batch xxx --sequence xxx --hidden xxx --output xxx --seed xxx`  
Explanation of the arguments: cluster --> if using gpu or not; data --> Carla driving data or LGSVL driving data; network --> which network to be trained; name --> file name to save results; epoch --> desired training epochs; batch --> the batch size; sequence --> time sequence length for LSTM/GRU/CT-GRU/NCP; hidden --> hidden state of LSTM/GRU/CT-GRU; output --> output dimensions,1 means predicting only steering, 3 means predicting steering, throttle, brake; seed --> random seed for neural network initialization.


## **Test the trained policies**

### Performance of different CNN head in combination with NCP
To compare different CNN heads, the absolute deviation is calculated, run  
`python calculate_deviation_cnn_head.py  --cluster yes`  
The latency of each combination of CNN+NCP is calculated, run  
`python latency_cnn_head.py`  

### Performance of trained networks by using CARLA data  

Sequential plot of commands is done to check the trained policy by using Carla driving data.
To plot the sequential commands, run  
`python sequential_plot_all.py  --cluster yes` for the models predicting steering, throttle and brake  
`python sequential_plot_steering.py  --cluster yes` for the models predicting only steering  

The latency of all neural networks is calculated, run
`python latency_all_models.py`  

### Performance of trained networks by using LGSVL data  
Online simulation in simulator LGSVL, script `ros_control.py`  aims to run in the ros package to drive the vehicle in simulator LGSVL.  


## **Results visualization**
Folder `Visualization` contains all the scripts for visualization.
To visualize the action distribution of Carla driving data, run  
`python Visualization/action_distribution_carla.py --cluster yes`  

To visualize the action distribution of LGSVL driving data, run  
`python Visualization/action_distribution_lgsvl.py --cluster yes`   

To plot histograms and boxplots of absolute deviation for CNN head comparison, run  
`python Visualization/deviation_analysis_cnn_head.py --cluster yes`   

To plot histograms and boxplots of absolute deviation for all 5 neural networks, run  
`python Visualization/deviation_analysis_all_models.py --cluster yes`  

To plot the loss curves of different CNN heads in combination with NCP, run  
`python Visualization/carla_loss_cnn_head.py --cluster yes --seed xxx`, possible seeds are 100, 150, 200.
 
To plot the loss curves of predicting steering, throttle, brake and loss curves of predicting steering in one plot, run  
`python Visualization/carla_loss_comparison.py --cluster yes --seed xxx`, possible seeds are 100, 150, 200.

To plot the loss curves of different neural networks in one plot, run  
`python Visualization/carla_loss_all_models.py --cluster yes --seed xxx`  for training on CARLA data  
`python Visualization/lgsvl_loss_all_models.py --cluster yes --seed xxx`  for training on LGSVL data  
Possible seeds are 100, 150, 200.


To plot the boxplots of absolute deviation in steering between models predicting steering, throttle, brake and models predicting only steering, run    
`python Visualization/all_vs_steering.py --cluster yes`  
 


### 
