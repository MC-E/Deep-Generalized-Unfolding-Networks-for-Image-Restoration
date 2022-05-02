## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
python train_plus.py
```

## Evaluation

### Download the [model](https://drive.google.com/file/d/1bitvtmJAE1iKpFmdGx3OrN6Xti0JRPLc/view?usp=sharing) and place it in ./pretrained_models/

#### Testing on GoPro dataset
- Download [images](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing) of GoPro and place them in `./Datasets/GoPro/test/`
- Run
```
python test.py --dataset GoPro
python test_plus.py --dataset GoPro
```

#### Testing on HIDE dataset
- Download [images](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK?usp=sharing) of HIDE and place them in `./Datasets/HIDE/test/`
- Run
```
python test.py --dataset HIDE
python test_plus.py --dataset HIDE
```

#### To reproduce PSNR/SSIM scores of the paper on GoPro and HIDE datasets, run this MATLAB script
```
evaluate_GOPRO_HIDE.m 
```
