# PROFIT
PROFIT implementation

# How to train 4-bit MobileNet-v2

1. download pre-trained model from pytorch repository

mkdir ./pretrained && wget -P ./pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

2. apply PROFIT with progressive quantization, teacher-student and weight ema

python train_ts.py --ckpt ./checkpoint/mobilenetv2 --model mobilenetv2 --teacher resnet101 --quant_op duq --use_ema --stabilize --w_bit 8 5 4 --a_bit 8 5 4 --w_profit 4 --decay 0.00004 --lr 0.005 
