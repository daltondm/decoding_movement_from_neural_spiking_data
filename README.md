# decoding_movement_from_neural_spiking_data

## Summary
#### Data
I use spiking data from 175 neurons to predict hand movements in 3D dimensions.
1. Neural Data: recorded from a Utah Array implanted in sensorimotor cortex of a common marmoset. Spike counts are binned in 20ms windows.
2. Kinematic Data: 3D hand position, which was estimated from video data using DeepLabCut. 
3. Experimental Behavior: the marmoset is capturing live moths in a prey-capture box. 

#### Models
I implement an LSTM model to predict instananeous position of the hand in 3D from spike count data in 175 neurons. I define a sequence length and decoder lead time such that the specified time bins of neural activity predict movement a short time later (50ms). There is also a linear readout layer after the LSTM layer to convert outputs from the LSTM to x-y-z position. 

#### Results
I improve the performance on held-out test data from r2=0.79 to r2=0.86 by:
1.  Doubling the size of the hidden layer(s).
2.  Adding a second hidden layer. 
3.  Regularization: dropout=0.4 and weight_decay=1e-5
4.  Decreased epochs required by increasing the learning rate from 0.001 to 0.01.

## Code

[collect_data_from_nwb.py](/collect_data_from_nwb.py): samples spikes and position from NWB file.

[decoding_movement_with_LSTMs.ipynb](/decoding_movement_with_LSTMs.ipynb): Implementation of LSTM model, with longer description of model development and evaluation. 