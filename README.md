# Global Wheat Detection
<p align="center"><img src="ProjetDL_images/desc.png" /></p>

## Description
Open up your pantry and you’re likely to find several wheat products. Indeed, your morning toast or cereal may rely upon this common grain. Its popularity as a food and crop makes wheat widely studied. To get large and accurate data about wheat fields worldwide, plant scientists use image detection of "wheat heads"—spikes atop the plant containing grain. These images are used to estimate the density and size of wheat heads in different varieties. Farmers can use the data to assess health and maturity when making management decisions in their fields.

However, accurate wheat head detection in outdoor field images can be visually challenging. There is often overlap of dense wheat plants, and the wind can blur the photographs. Both make it difficult to identify single heads. Additionally, appearances vary due to maturity, color, genotype, and head orientation. Finally, because wheat is grown worldwide, different varieties, planting densities, patterns, and field conditions must be considered. Models developed for wheat phenotyping need to generalize between different growing environments. Current detection methods involve one- and two-stage detectors (Yolo-V3 and Faster-RCNN), but even when trained with a large dataset, a bias to the training region remains.

The Global Wheat Head Dataset is led by nine research institutes from seven countries: the University of Tokyo, Institut national de recherche pour l’agriculture, l’alimentation et l’environnement, Arvalis, ETHZ, University of Saskatchewan, University of Queensland, Nanjing Agricultural University, and Rothamsted Research. These institutions are joined by many in their pursuit of accurate wheat head detection, including the Global Institute for Food Security, DigitAg, Kubota, and Hiphen.

In this competition, you’ll detect wheat heads from outdoor images of wheat plants, including wheat datasets from around the globe. Using worldwide data, you will focus on a generalized solution to estimate the number and size of wheat heads. To better gauge the performance for unseen genotypes, environments, and observational conditions, the training dataset covers multiple regions. You will use more than 3,000 images from Europe (France, UK, Switzerland) and North America (Canada). The test data includes about 1,000 images from Australia, Japan, and China.

Wheat is a staple across the globe, which is why this competition must account for different growing conditions. Models developed for wheat phenotyping need to be able to generalize between environments. If successful, researchers can accurately estimate the density and size of wheat heads in different varieties. With improved detection farmers can better assess their crops, ultimately bringing cereal, toast, and other favorite dishes to your table.

## Data
More details on the data acquisition and processes are available at https://arxiv.org/abs/2005.02162

**What should I expect the data format to be?**\
The data is images of wheat fields, with bounding boxes for each identified wheat head. Not all images include wheat heads / bounding boxes. The images were recorded in many locations around the world.

The CSV data is simple - the image ID matches up with the filename of a given image, and the width and height of the image are included, along with a bounding box (see below). There is a row in train.csv for each bounding box. Not all images have bounding boxes.
Most of the test set images are hidden. A small subset of test images has been included for your use in writing code.

**What am I predicting?**\
You are attempting to predict bounding boxes around each wheat head in images that have them. If there are no wheat heads, you must predict no bounding boxes.

**Files**
* train.csv - the training data
* sample_submission.csv - a sample submission file in the correct format
* train.zip - training images
* test.zip - test images

**Columns**
* image_id - the unique image ID
* width, height - the width and height of the images
* bbox - a bounding box, formatted as a Python-style list of [xmin, ymin, width, height]
* etc.

## Metrics
This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a set of predicted bounding boxes and ground truth bounding boxes is calculated as:
<p align="center"><img src="ProjetDL_images/m1.png" /></p>

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.75 with a step size of 0.05. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.
At each threshold value t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:
<p align="center"><img src="ProjetDL_images/m2.png" /></p>

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object.

Important note: if there are no ground truth objects at all for a given image, ANY number of predictions (false positives) will result in the image receiving a score of zero, and being included in the mean average precision.
The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold:
<p align="center"><img src="ProjetDL_images/m3.png" /></p>

In your submission, you are also asked to provide a confidence level for each bounding box. Bounding boxes will be evaluated in order of their confidence levels in the above process. This means that bounding boxes with higher confidence will be checked first for matches against solutions, which determines what boxes are considered true and false positives.
Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

**Intersection over Union (IoU)**\
Intersection over Union is a measure of the magnitude of overlap between two bounding boxes (or, in the more general case, two objects). It calculates the size of the overlap between two objects, divided by the total area of the two objects combined.
It can be visualized as the following:
<p align="center"><img src="ProjetDL_images/m4.png" /></p>
The two boxes in the visualization overlap, but the area of the overlap is insubstantial compared with the area taken up by both objects together. IoU would be low - and would likely not count as a "hit" at higher IoU thresholds.


## Concepts
* **MixUp**\
MixUp is a recently proposed method for training deep neural networks(DNN)
where additional samples are generated during training by convexly combining
random pairs of images and their associated labels. While simple to implement,
it has been shown to be a surprisingly effective method of data augmentation
for image classification: DNNs trained with mixup show noticeable gains in
classification performance on a number of image classification benchmarks.
Mixup training is based on the principle of Vicinal Risk Minimization (VRM): the classifier
is trained not only on the training data, but also in the vicinity of each training sample. 
MixUp can be represented with this simple equation:

<p align="center"><b>newImage = alpha * image1 + (1-alpha) * image2</b></p>

This newImage is simply a blend of 2 images from your training set, it is that simple! So, what will be the target value for the newImage?

<p align="center"><b>newTarget = alpha * target1 + (1-alpha) * target2</b></p>

The important thing here, is that you don’t always need to One Hot Encode your target vector. In case you are not doing OneHotEncoding, custom loss function will be required.

* **FixMatch**\
FixMatch is a semi-supervised learning method that use consistency regularization as cross-entropy between one-hot pseudo-labels of weakly translation applied images and prediction of strongly translated them. It is possible to learn with even a very small amount of labeled data.\
Semi-supervised learning (SSL) is a learning method where learning is performed with a small number of labeled data and a large number of unlabeled data.The biggest advantage against supervised learning is that you do not need to prepare labels for all data.
<p align="center"><img src="ProjetDL_images/fixmatch1.png" /></p>

<p align="center"><img src="ProjetDL_images/fixmatch.png" /></p>

 
For more understanding about FixMatch see : https://amitness.com/2020/03/fixmatch-semi-supervised/ <br> 
https://medium.com/analytics-vidhya/fixmatch-semi-supervised-learning-method-that-can-be-learned-even-if-there-is-only-one-labeled-e7e1b37e8935


* **TTA**
Similar to what Data Augmentation is doing to the training set, the purpose of Test Time Augmentation(TTA) is to perform random modifications to the test images. Thus, instead of showing the regular, “clean” images, only once to the trained model, we will show it the augmented images several times. We will then average the predictions of each corresponding image and take that as our final guess.
