## CapsNet-1D
1D Capsule Network for Real-Time Gesture Recognition.

## Usage 

#### Pretraining

Clone or Download the repository and save the files in your Python directory. Pretraining assumes that you have a custom dataset of gesture activity with rows as signals and columns as features.


Clean, preprocess and calibrate your dataset using sensor corrections in clean.py.

```
python3 clean.py
```

Pass your dataset as the input to the 'caps_net.py' file and then run the code.

```
python3 pretrain_capsnet.py
```

#### Real-Time

Connect and place the GY-80 real time sensor on the forearm of human subject. Launch the interface using `gear_v2` GUI.

```
python3 gear_v2.py
```

Record and recognize gestures using sensor readings on-the-fly.

```
python3 real_time.py
```

## Dependencies 

```
python3 (3.6 <=)
keras (2.2.0 <=)
tensorflow v1 (1.10.0 <=)
numpy
```

## Acknowledgment
We acknowledge funding support from the Science & Engineering Research Board, Department of Science & Technology (DST), Government of India, SERB file number ECR/2016/000637.

