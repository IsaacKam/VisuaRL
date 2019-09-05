# VisuaRL: Code for K2VS, RLINE, ASRL and the Segmenter
This is part of my Msc Machine Learning thesis at UCL titled:  ' Leveraging Visual priors for deep reinforcement learning'

Our models are compatible with any of the Atari Gym environments. We use Pong as an example in the bash scripts below.

## Segmenter Intructions:
This is the preset model
```bash
python3 main.py --env-name "PongNoFrameskip-v4"  --num-processes 45 --log-dir test 
```
## K2VS Intructions:
Add --key-value 1 to your bash script
```bash
python3 main.py --env-name "PongNoFrameskip-v4"  --num-processes 45 ---key-value 1 --log-dir test 
```
## RLINE Intructions:
comment line 26 on model.py out and uncomment line 27
```bash
python3 main.py --env-name "PongNoFrameskip-v4"  --num-processes 45 log-dir test 
```
## ASRL Intructions:
change branch to RLDO and run.
```bash
python3 main.py --env-name "PongNoFrameskip-v4"  --num-processes 45 --priors 3 --key-value 1 --log-dir test  --game Pong 
```

## Choosing your visual priors intructions:
Go into segment.py and uncomment the block that corresponds with the visual prior you want.
Some priors have two options (priors 1 or prior s2), you can select your choice by adding --priors 1/2 into your script.

## Contact me!
If you have any questions feel free to email me at zcahika@ucl.ac.uk
