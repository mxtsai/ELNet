# Implementation of Efficiently Layered Network (ELNet) 


Based on *Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)* by [Maxwell Tsai](https://mxtsai.github.io/)

Please see `model_elnet.py` for the PyTorch implementation of ELNet.

[Paper Link](https://arxiv.org/abs/2005.02706) / [5-min Video Presentation](https://www.youtube.com/watch?v=ucWYdEJ545k) / [Teaser](https://www.youtube.com/watch?v=8nO-E_2aNcE)

## Network Architecture
<img src='https://raw.githubusercontent.com/mxtsai/ELNet/master/ELNet_architecture.png' align="right" width=320>

The three main components of ELNet are:
  1. Block Modules *(in purple)*
  2. Multi-Slice Normalization *(in green)*
  3. BlurPool operations *(in yellow)*
  
Block Modules are designed to introduce more non-linearities in the network, and they may be repeated while ensuring equal input and output dimension. Multi-Slice normalization allows for slice-independent normalization of the feature representations in the network. BlurPool downsampling ensures anti-aliased represenations during pooling operations. Please check the paper for more details.

### Input Dimesion
ELNet takes in a 3D input image of dimension `1 x S x H x W` where `S` is the number of slices of the image, and `H,W` are the spatial height and width of the image. (In the paper, `H,W = 256` and `S` varies between cases to case)

### Hyperparameters

* `K` - A parameter that controls the channel dimension of the feature representations in the network (see diagram on the right). The output of the ELNet feature extractor is a feature vector of dimension `16K`. The model size grows quadratically as a function of `K`, so it is recommended to adjust `K` first according to the model size desired (see paper for detail).

* `norm_type` - The type of multi-slice normalization desired throughout the network. Options include `layer`,`instance`,`batch` for layer normalization, instance normalization, and batch normalization. Adjust this parameter according to the imaging plane of the input image (e.g. `layer` for axial imaging plane, and `instance` for coronal imaging plane).

* `aa_filter_size` - The kernel size for [BlurPool](https://github.com/adobe/antialiased-cnns) (anti-aliasing) downsampling. Adjustments to `aa_filter_size` will affect downsampled feature representations (`aa_filter_size` was kept to 5 in the paper).

* `num_classes` - The number of classes to perform classification (default is 2). 

* `weight_init_type` - The type of weight initialization to initialize the weights of ELNet with. Options include `normal` and `uniform`. 

* `seed` - The random seed for deterministic results (useful for debugging).

## Questions & Citation

Feel free to contact me if there are any questions or comments regarding the paper or the implementation. 

If you find this useful, you are welcome to cite our work using:
```
@InProceedings{pmlr-v121-tsai20a, 
	title = {Knee Injury Detection using MRI with Efficiently-Layered Network (ELNet)}, 
	author = {Tsai, Chen-Han and Kiryati, Nahum and Konen, Eli and Eshed, Iris and Mayer, Arnaldo}, 
	pages = {784--794}, 
	year = {2020}, 
	volume = {121}, 
	series = {Proceedings of Machine Learning Research}, 
	publisher = {PMLR}, 
}
```
