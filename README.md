# VisuaRL
Code for  K2VS, RLINE, ASRL and segmenter. This is part of my Msc Machine Learning thesis at UCL titled:  ' Leveraging Visual 
## Segmenter Intructions:
This is the preset model
## K2VS Intructions:
Add --key-value 1 to your bash script
## RLINE Intructions:
comment line 26 on model.py out and uncomment line 27
## ASRL Intructions:
change branch to RLDO and run.

```bash
python3 ./main.py --env-name "PongNoFrameskip-v4"  --num-processes 45 --priors 1 --key-value 1 --log-dir logs_spare/Pong_folder/ASRL_SEG/Segm_no_mult_kv-0 --game Pong 
```
