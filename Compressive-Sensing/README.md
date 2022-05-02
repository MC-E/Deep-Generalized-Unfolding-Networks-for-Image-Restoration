## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

### For DGUNet
```
Step 1: python train.py
Step 2: python train_deblock.py
```

### For DGUNet_plus
```
Step 1: python train_plus.py
Step 2: python train_plus_deblock.py
```

## Evaluation

### Download the [model](https://drive.google.com/file/d/1euV8SnXHuYswwbm9FaV1y5RQRLd58ou_/view?usp=sharing) and place it in ./model/

- Download [images](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing) of Set11&BSD68 and place them in `./Datasets/`
- Run
```
python test_deblock.py
python test_plus_deblock.py
```
