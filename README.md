# Intel-Scene-Classification
Solution of Intel Scene Classification challenge getting accuracy of 94.2 %
![image](images/Capture5.PNG)
[Competition Link](https://datahack.analyticsvidhya.com/contest/practice-problem-intel-scene-classification-challe/)



[Colab Link](https://colab.research.google.com/drive/1NggCLuuEJnxQo7DVThrDUW1yjEQn3XaN#scrollTo=xJUjD9s5g8s5)


The Intel scene classification problem contained around 20000 images belonging to 6 classes i.e buildings, forests, glacier, mountain, sea and street. The challenge is to classify the images with more than 90% accuracy. The data for this challenge can be found [here](https://www.kaggle.com/nitishabharathi/scene-classification).

Following are the classes and labels of the images given in the challenge:



```
'buildings' -> 0
'forest' -> 1
'glacier' -> 2
'mountain' -> 3
'sea' -> 4
'street' -> 5
```

The labels for the images are given in a csv file (for both train and test) which contains two colums, one for the image name and other for the label of the image as shown below:

| Variable	| Definition |
| ------------- | ----------------- |
| image_name	| Name of the image in the dataset (ID column) |
| label | Category of natural scene (target column) |
 
 
To solve this challenge we have utilised [Fastai vision](https://docs.fast.ai/vision.html) library which contains all the necessary funtions for computer vision tasks and allows us to quickly train and test the model with best parameters.
#### Databuch creation:
To use fastai library first task is to create a databunch from the images so that we can quickly load the images for training and testing in batches. For this purpose we'll use Fastai's [ImageDataBunch class](https://github.com/fastai/fastai/blob/master/fastai/vision/data.py#L85). Since the image details are given in csv format, we'll use the [from_csv](https://github.com/fastai/fastai/blob/master/fastai/vision/data.py#L123) method of ImageDataBunch class.

#### Image Augmentation:
For image augmentations we'll use Fastai's [get_tranforms](https://github.com/fastai/fastai/blob/master/fastai/vision/transform.py#L307) class. By defauld get_transform function applies 8 types of transformations to the images i.e crop_padding, flipping, symmetric warping, rotating, zooming, brightness, contrast and again crop_padding. In our case rotating the images may cause confusion so we'll remove the rotation and increase the zooming of images from 10% to 50%.
``` python
tfms=get_transforms(max_rotate=0,max_zoom=1.5)
bunch=ImageDataBunch.from_csv('.',csv_labels='train.csv',folder='./train',size=150,ds_tfms=tfms,valid_pct=0.2,test='./test')
```
In above code we're creating a Imagedatabunch with training images in ./train folder and test images in ./test folder with augmentation. We are resizing all the images to 150X150 for consistent imput to the model.

#### Convolutional Neural Network Model:
we'll use Fastai's [cnn_lerner](https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py#L90) class to build the model using pretrained resnet18 model.
```python
learn = cnn_learner(bunch, models.resnet18, metrics=accuracy)
```
#### Learning rate finder:
Let's use Fastai lr finder class to find the best learning rate for the model:

<img src="images/lr.PNG" width="500">

From above image we can see that the loss is minimum when using the learning rate of 0.001. So we'll train the model for 20 epochs with the learning rate 0.001. Lets plot the loss after training the images:

<img src="images/loss.PNG" width="500">

Lets plot the confusion matrix for the predicted images:

<img src="images/Capture1.PNG" width="300">

Lets plot some images and seet the labels. Im following images, the labels are in order of prediction/actual/loss/probability.

<img src="images/Capture3.PNG" width="800">

Lets upload the results on the Analytic vidya wesite and see the ranking:

<img src="images/Capture.PNG" width="800">

