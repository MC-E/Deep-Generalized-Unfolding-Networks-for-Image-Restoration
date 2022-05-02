
## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
python train_plus.py
```


## Evaluation

1. Download the [model](https://drive.google.com/file/d/1bitvtmJAE1iKpFmdGx3OrN6Xti0JRPLc/view?usp=sharing) and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800) from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
```
python test.py
python test_plus.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
evaluate_PSNR_SSIM.m 
```
