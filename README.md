
![Alt text](/images/logo.png "Halo Diagram")
# A library That Uses Quantized Diffusion Model With Clustered Weights For Efficiently Generating More Image Datasets On-Device.

# AUTHORS
Henry Ndubuaku\
ndubuakuhenry@gmail.com

# BACKGROUND
Every machine learning project have to first overcome the issue of dataset availability. This however require a lot of expertise to navigate. For classification and image understanding problems, augmentation techniques like flipping, cropping, etc. For supervised learning tasks, annotation tools like GCP Labelling Services and AWS Mechanical Turks come in handy. Albeit, for simply generating more images, it gets more technical. 

# APPROACH
Halo uses diffusion which yields exceptionally crisp images (). Vision Transformer blocks are then sandwitched between each residual block for scaling the parameters. This model is then pre-trained on a large collection of image dataset. 

The resulting model's weights are then clustered to reduce the number of parameters and the models size. Clustering, or weight sharing, reduces the number of unique weight values in a model, leading to benefits for deployment. It first groups the weights of each layer into N clusters, then shares the cluster's centroid value for all the weights belonging to the cluster. The model is fine-tuned for a few epochs.

Next, the parameters of are quantized which involves reducing the precision of the weights, biases, and activations such that they consume less memory. The model is one again fine-tuned for a few epochs.

Finally, a TFLite version of the model which runs on edge devices, is packaged into a library.

# EXPERIMENTATION AND RESULTS
A small version model with only 15m parameters was trained on CelebA. Below are some synthethic samples, when adapted to various ethnicities.

![Alt text](/images/asian.png "results")
![Alt text](/images/blacks.png "results")
![Alt text](/images/blonde.png "results")

When trained at scale for every kind of image, granular results like speed increments, size reduction, and metrics will be carefully studied and analysed.

# USAGE (BETA)
1. This requires "tensorflow-GPU>=2.11.0" and "tensorflow_model_optimization" for now.
2. Download the model weights [here](https://drive.google.com/drive/folders/18MzoJbm9qSmGaE1-pwPnwqd85LvXJKve?usp=sharing)
3. Place inside the parent HALO folder.
4. Adapt to your dataset and generate more samples with the code below.
```
    from halo.tuner import tune, generate

    model = tune("path_to_folder_of_images", "path_to_checkpoint")
    generate(model=model, n_samples=10, path="my_path", extension=".jpg")
```

# HOW TO HELP
The code for quantizing and clustering the weights of the model is actively being tested and modified. On completion, the model will be scaled and pre-trained on massive dataset, as well as thouroughly tested. Afterwards, it will miniaturized and converted to a TFLite version. A CLI would be developed and the whole package deployed to PyPI. Contribution to any of these will be highly beneficial. Please send the author an email.

# END GOAL
ML practitioners will simply install the library with "pip install halo", adapt to their dataset in a script with "halo.tuner.tune()", then generate samples with "halo.generate()". They can also use via command line with "halo path_to_dataset path_to_output".

# SPONSORSHIP 
This authour works full-time and the cost of massively training halo would be expensive. Funds recieved will be reserved for computing expenses. Excess funds would get the author onboard full-time. Even more funding would get more hands on deck.
