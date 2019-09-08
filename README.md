# Transformer_CycleGAN_Text_Style_Transfer-pytorch
Implementation of CycleGAN for Text style transfer with PyTorch.

## CycleGAN Architecture
![](https://i.imgur.com/tHl26oG.png)

## Model Detail
![](https://i.imgur.com/UrLR9qS.png)
![](https://i.imgur.com/tmMivIm.png)

- We simply use softmax weighted sum over all word vectors to overcome the discrete gradient issue in GAN training process.
