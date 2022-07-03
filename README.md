# Overview
This is the official code for "Accurate and Clear Quantitative Precipitation Nowcasting Based on a Deep Learning Model with Consecutive Attention and Rain-Map Discrimination".

## How to train the model:
We, in our codebase, process the rain and the radar data and store them as monthly .pkl files: one file for one month.
Our training module assumes these .pkl files to be present at `DATA_PKL_DIR` variable present in core/constants.py. Please update the variable appropriately before starting the training procedure. In case the .pkl files have not been created, please give an empty directory in that variable. The training procedure will create the .pkl files and will fill them there. 

### How to create the monthly pre-processed .pkl files.
.pkl files are created by us to speed up the data loading process at the beginning of the training/evaluation procedure.
There are two steps involved in their creation.
#### How should be the input look like:
Radar files should be present in the nested directory structure: `RADAR_RAW_DIR/YYYY/YYYYMM/YYYYMMDD/filename.nc`. One example file could be `/tmp2/AccClearQPN/data/RADAR/2018/201805/20180507/MREF3D21L.20180507.0000.nc`. Rain files should again be present in the identical structure. The structure is assumed to be inside `RAIN_RAW_DIR`. 

#### Steps to create the .pkl files.
1. Run `python core/compressed_radar_data.py $RADAR_RAW_DIR /tmp2/ashesh/AccClearQPN/data/compressed/`. Here, the first argument is the directory where raw RADAR data is present. The second argument is the place where the compressed data is to be stored. Same needs to be done for rain data via `core/compressed_rain_data.py`. 


This essentially compresses the 10 min raw radar and rain maps. The idea is that most pixels in these maps are 'empty'. So, it makes sense to keep the data in a compressed way where we keep a list of three tuples: x and y position in the map and the value. The (x,y) positions which are not present are assumed to be zero valued. 
2. Collect the 10 minute compressed radar and rain maps into a monthly .pkl file. Nothing needs to be done explicitly for this. This is automatically done when starting the training for the first time. From the next time, data is directly loaded from .pkl files.


### How to start the training.
This code works on linux based operating system. (Ubuntu 16.04.7 LTS specifically)
1. Set the environment variable which stores where your data is located. For ex: `export ROOT_DATA_DIR=/tmp2/data/precipitation/`
2. Go to the code repository directory. For ex: `cd /home/ubuntu/code/AccClearQPN`
3. Set the PYTHONPATH on the code directory: `export PYTHONPATH=/home/ubuntu/code/AccClearQPN`
4. Run the following command.
```
python scripts/pl_run.py --batch_size=32 --train_start=20180401 --train_end=20180630 --val_start=20180701 --val_end=20180731  --data_kwargs=sampling_rate:5
```
One needs to give the `train_start` and other dates in accordance to the data one has in `ROOT_DATA_DIR`.

