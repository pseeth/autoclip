# AutoClip
Pytorch and tensorflow implementations (and variations) of the AutoClip gradient smoothing procedure from [Seetharaman et al](https://arxiv.org/abs/2007.14469).

> Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux. "AutoClip: Adaptive Gradient Clipping for Source Separation Networks." 2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2020.

## About

While training your model, AutoClip keeps a running history of all of your model's gradient magnitudes. Using these, the gradient clipper can adaptively clamp outlier gradient values before they reach the optimizer of your choice.

While AutoClip is great as a preventative measure against exploding gradients, it also speeds up training time, and encourages the optimizer to find more optimal models. At an intuitive level, AutoClip compensates for the stochastic nature of training over batches, regularizing training effects.

## Installation

AutoClip is listed on pypi. To install AutoClip simply run the following command
```
pip install autoclip
```
and the `autoclip` package will be installed in your currently active environment.

## Torch API

Below are some examples how to use `autoclip`'s torch API.

### Clippers as Optimizer Wrappers
Using the optimizer wrapping pattern is the recommended way to use AutoClip, and `autoclip`'s torch API supports wrapping arbitrary pytorch optimizers. The wrapping pattern allows you to avoid changing your training code when you want to use an AutoClip clipper. This is especially useful if you do not own the training code for whatever reason. (Say for example you are using someone else's Trainer class, as is often the case with frameworks like `huggingface`.)

The following is an example of how to integrate AutoClip into your model training using this pattern:
```python
import torch
from autoclip.torch import QuantileClip

model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 2)
)

optimizer = torch.optim.AdamW(model.parameters())
optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=0.9, history_length=1000)
```
Now you can use the optimizer just like you would have before adding the clipper, and the clipping will be applied automatically.

### Raw AutoClip Clippers
You can still use the clipper manually if you would like. If this is the case, then you would create your clipper like this:
```python
import torch
from autoclip.torch import QuantileClip

model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 2)
)

clipper = QuantileClip(model.parameters(), quantile=0.9, history_length=1000)
```
Then, to clip the model's gradients, simply run the clipper's `.step()` function during your training loop. Note that you should call the clipper's `step` before you call your optimizer's `step`. Calling it after would mean that your clipping will have no effect, since the model will have already been updated using the unclipped gradients. For example:
```python
for batch_num, batch in enumerate(training_dataset):
    model_prediction = model(batch['data'])
    loss = loss_function(model_prediction, batch['targets'])
    loss.backward()
    clipper.step() # clipper comes before optimizer
    optimizer.step()
```

### Global vs Local Clipping
`autoclip`'s torch clippers support two clipping modes. The first is `global_clipping`, which is the original AutoClip as described in Seetherman et al. The second is local or parameter-wise clipping. In this mode a history is kept for every parameter, and each is clipped according to its own history. By default, the `autoclip` clippers will use the parameter-wise clipping.
To use the global mode, simply pass the appropriate flag:
```python
clipper = QuantileClip(
    model.parameters(),
    quantile=0.9,
    history_length=1000,
    global_clipping=True
)
```

### Checkpointing
The torch clippers also support checkpointing through `state_dict()` and `load_state_dict()`, just like torch models and optimizers. For example, if you want to checkpoint a clipper to `clipper.pth`:
```python
clipper = QuantileClip(model.parameters())
torch.save(clipper.state_dict(), 'clipper.pth')

# Then later
clipper = QuantileClip(model.parameters())
clipper.load_state_dict(torch.load('clipper.pth'))
```
Keep in mind that just like a torch optimizer this will error if you give the clipper differently sized model parameters.

While it is generally recommended to use `state_dict`s instead (see the [pytorch documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model) on this subject for more info), you may also use `torch.save` and `torch.load` directly to pickle the entire clipper object.

## Tensorflow
`autoclip`'s tensorflow API does not currently have feature parity with the torch API (If you want to change this, feel free to [contribute](https://github.com/HesitantlyHuman/autoclip/issues/2)).
As it is, the tensorflow API currently only supports the original AutoClip algorithm, and does not support checkpointing. Below is a short example:
```python
import tensorflow as tf
from autoclip.tf import QuantileClip

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(50),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(
            2,
            activation=tf.keras.activations.tanh
        ),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        gradient_transformers=[
            QuantileClip(
                quantile=0.9,
                history_length=1000
            )
        ]
    ),
    loss="mean_absolute_error",
    metrics=["accuracy"],
)
model.fit(train_data, train_targets)
```
