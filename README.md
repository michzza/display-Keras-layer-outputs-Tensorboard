# display-Keras-layer-outputs-Tensorboard
A Python script where we build a simple Keras sequential model, and define a custom callback that allows to display in Tensorboard intermediate layer outputs during training.

Outputs of the model's intermediate layers per epoch are saved to Tensorboard. Once the script is run, they can be displayed using the following command 

```
$ tensorboard --logdir path/to/dir/tensorboard
```

This is an example of the ouputs for the first 2 layers:
![image](https://user-images.githubusercontent.com/78221892/147249961-0e2fc821-c8c1-4109-bcb3-f1dab0edbe08.png)
