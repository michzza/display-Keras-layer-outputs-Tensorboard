# display-Keras-layer-outputs-Tensorboard
A Python script where we build a simple Keras sequential model, and define a custom callback that allows to display in Tensorboard intermediate layer outputs during training.

Outputs of the model's intermediate layers per epoch are saved to Tensorboard. Once the script is run, they can be displayed using the following command 

```
$ tensorboard --logdir path/to/dir/tensorboard
```

This is an example of the ouputs for the first layer:
![image](https://user-images.githubusercontent.com/78221892/147237171-1515ce6f-ea26-480a-a665-0e81c14ca3b4.png)
