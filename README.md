# AutoClip: Adaptive Gradient Clipping

This repository accompanies the [paper](https://arxiv.org/abs/2007.14469):

> Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux. "AutoClip: Adaptive Gradient Clipping for Source Separation Networks." 2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2020.

At the moment it contains a [sample implementation of AutoClip](autoclip.py) that can be integrated into an ML project based on PyTorch easily.
Soon it will come as a Python package that can be installed and attached to a training script more easily.

## Abstract
> Clipping the gradient is a known approach to improving gradient descent, but requires hand selection of a clipping threshold hyperparameter. We present AutoClip, a simple method for automatically and adaptively choosing a gradient clipping threshold, based on the history of gradient norms observed during training. Experimental results show that applying AutoClip results in improved generalization performance for audio source separation networks. Observation of the training dynamics of a separation network trained with and without AutoClip show that AutoClip guides optimization into smoother parts of the loss landscape. AutoClip is very simple to implement and can be integrated readily into a variety of applications across multiple domains.

## Presentation

This work was presented at MLSP2020 in a special session. If you missed my talk, no worries, there's a pandemic happening so it's recorded! Here it is: 
https://share.descript.com/view/18725e02-95fe-4fb0-b32d-26c63617d482.

## Citation
```
@inproceedings{seetharaman2020autoclip,
  title={AutoClip: Adaptive Gradient Clipping for Source Separation Networks},
  author={Seetharaman, Prem, and Wichern, Gordon, and Pardo, Bryan, and Le Roux, Jonathan},
  booktitle={2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2020},
  organization={IEEE}
}
```


## Training dynamics

### Mask-inference loss

![](images/mi.gif)

### Whitened K-Means loss

![](images/wkm.gif)

Training dynamics of a smaller mask inference network (2 BLSTM layers with 300 hidden units) with mask-inference loss and whitened k-means loss, with and without AutoClip. The top left figure shows the norm of the step size taken on the model parameters. The top right figure shows the training loss over time, showing that AutoClip leads to better optimization. The bottom figures show the relationship between gradient norm and a measure of smoothness along the training trajectory. Statistics were recorded every 20 iterations during training.  With AutoClip, we observe a stronger correlation (r-value of .86), compared to without (r-value of .62). All gradients to the right of the dashed black line in the bottom right plot are clipped. We show the location of the AutoClip threshold at the end of training. The threshold changes during training.
