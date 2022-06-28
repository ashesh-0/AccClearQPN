# Overview
This is the official code for "Accurate and Clear Quantitative Precipitation Nowcasting Based on a Deep Learning Model with Consecutive Attention and Rain-Map Discrimination".

Example:
```
python scripts/pl_run.py --gpus=-1 --loss_kwargs=type:13 --model_kwargs=type:BalancedGRUAdverserial --data_kwargs=type:17
```
Refer to `model_type.py` for `type` argument of `model_kwargs`. 

