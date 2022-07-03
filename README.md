# Overview
This is the official code for "Accurate and Clear Quantitative Precipitation Nowcasting Based on a Deep Learning Model with Consecutive Attention and Rain-Map Discrimination".

## How to train the model:
We, in our codebase, process the rain and the radar data and store them as monthly .pkl files: one file for one month.
This is done in the first run. Subsequent runs for the training simply loads the .pkl files and work with it.

### How to create the monthly pre-processed .pkl files.

### How to start the training.
This code works on linux based operating system. (Ubuntu specifically)
```
python scripts/pl_run.py --gpus=-1 --loss_kwargs=type:13 --model_kwargs=type:BalancedGRUAdverserial --data_kwargs=type:17
```
Refer to `model_type.py` for `type` argument of `model_kwargs`. 

