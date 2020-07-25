# AutoClip: Adaptive Gradient Clipping

This repository accompanies the forthcoming paper:

> Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux. "AutoClip: Adaptive Gradient Clipping for Source Separation Networks." 2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2020.

```
@inproceedings{seetharaman2020autoclip,
  title={AutoClip: Adaptive Gradient Clipping for Source Separation Networks},
  author={Seetharaman, Prem, and Wichern, Gordon, and Pardo, Bryan, and Le Roux, Jonathan},
  booktitle={2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2020},
  organization={IEEE}
}
```
At the moment it contains a sample implementation of AutoClip that can be integrated into an ML project based on PyTorch easily.
Soon it will come as a Python package that can be installed and attached to a training script more easily.

## Training dynamics

### Mask-inference loss

![](images/mi.gif)

### Whitened K-Means loss

![](images/wkm.gif)

> Training dynamics of a smaller mask inference network (2 BLSTM layers with 300 hidden units) with $\mathcal{L}_{\text{MI}}$, with and without AutoClip. The top left figure shows the norm of the step size taken on the model parameters. The top right figure shows the training loss over time, showing that AutoClip leads to better optimization. The bottom figures show the relationship between gradient norm and a measure of smoothness along the training trajectory. Statistics were recorded every 20 iterations during training.  With AutoClip, we observe a stronger correlation (r-value of $.86$), compared to without (r-value of $.62$). All gradients to the right of the dashed black line in the bottom right plot are clipped. We show the location of the AutoClip threshold at the end of training. The threshold changes during training.
