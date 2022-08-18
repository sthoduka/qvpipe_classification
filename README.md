This repository contains the code for the submission to the [ICPR VideoPipe Challenge - Track on Video Defect Classification](https://codalab.lisn.upsaclay.fr/competitions/2232).


The main requirements are:

```
pytorch==1.9.0
pytorch-lightning==1.6.1
torchnet==0.0.4
tensorboard==2.8.0
torchvision==0.10.0
opencv-python==4.5.5.62
```

To train the network, set parameters in `train.sh` and run it. To produce the results on the test set, run `test.sh` after specifying which checkpoint to use.


A short report can be found [on this page](https://videopipe.github.io/results/index.html) (see under Task 1: Video Defect Classification).
