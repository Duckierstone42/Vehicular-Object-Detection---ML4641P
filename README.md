# ML4641Project

## Introduction/Background

For our project we wish to apply object-detection to the field of self-driving, specifically by implementing machine learning models that can detect vehicles, specifically cars, as shown in our [dataset](https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset). We indicated that we were going to use a different dataset in the project proposal, but we ended up changing to this dataset. This dataset includes around 500 training images as well as around 100 testing images, where images come with a corresponding set of labels which indicate the size, position, and type of objects located in the image. If we end up needing more data, particularly to train some of the larger deep learning models we plan on using other datasets like the [car tracking and object detection dataset](https://www.kaggle.com/datasets/trainingdatapro/cars-video-object-tracking), which contains similar images, albeit at a different angle, perhaps allowing our models to better generalize. We currently plan on finetuning a variety of state-of the art object detection models [1], such as YOLO-v7 [2], and have looked at how others have applied such models onto the problem of vehicle detection [3].

## Problem Definition

With self-driving cars becoming more and more relevant in current society, it is crucially important that they are able to detect other vehicles to prevent collisions in a reliable, quick manner. We seek to explore and expand upon existing methods of using machine learning techniques to detect vehicles.

## Methods

Due to the relatively low amount of data, we plan on increasing the amount and diversity of the data using data augmentation techniques such as color jittering, gaussian noise, or blurring. We will also normalize our data, changing the intensity of the pixels to a more consistent distribution. Finally we will resize all the images to a common size, to ensure images have a consistent input size.

Finally, for our supervised techniques, we plan on training a plain neural network on the object-detection dataset, just to gain a minimum baseline on which more advanced models can improve on. We then plan on fine-tuning state of the art object detection models, such as YOLO or R-CNN. Finally, we will look at more novel object detection methods, such as those involving ViTs (Vision Transformers). As these models require lots of computational power to train, we plan on training them via either google colab or PACE-ICE.

### Implemented Method

We currently have fine-tuned a small YOLO-v5 model on our dataset, with all the code present to do so in test.ipynb. We used Batch Gradient Descent over 20 epochs, using the Adam optimizer with a learning rate of .0001. When we were pre-processing data, we performed data augmentation via color jittering and gaussian blurring through pytorch transforms, then ensured each image was 640 by 640 as input into the neural network.

#### Color Jittering

Color jittering is applied via the transforms.ColorJitter torch transform, with a brightness of .2, a contrast of .2, a saturation of .2, and a hue of .1. These values determine how much exactly the brightness, contrast, saturation, and hue are randomly jittered. 

#### Gaussian Blurring

Gaussian Blurring was applied via the transforms.GaussianBlur torch transform, with a kernel_size of (5,9) and sigma values of (.1,5). The sigma values define the standard deviation used for creating the kernel to perform the blurring. The gaussian kernel is then just convolved with the data in a similar matter to the convolutions in a convolutional neural network, or like in non-parameteric density estimation methods.

## Results and Discussion

We plan to use accuracy, latency, as well as mAP (mean average precision), a benchmark often used for detection problems. While it may be unrealistic to achieve in the tight timeframe of this project, we hope to be able to create/fine-tune object detection models capable of detecting vehicles with quantitative metrics at the level other papers have achieved on the same problem. For example, [3] was able to train a YOLOv3 model to achieve a mAP of 72.8. Realistically though, we hope to be able to train/finetune an object-detection model at a minimum mAP of around 20-30, a base value which should hopefully indicate that our model is able to identify vehicles with reasonable accuracy.

### Visualizations

![image](./images/train_image.png)

The above image is an example image from the training data set. The borders around each object are called the "bounding boxes", and we aim to train our model to construct boxes just like these. The green boxes present above represent the labels given in the dastaset.

![alt text](./images/image.png)
This is an image that our model has been evaluated on, with bounding boxes constructed for each object detected, as well as their associated confidence levels (between 0 and 1).

The red boxes show the predicted bounding boxes while the green boxes show the actual labels. You can clearly see that the object detection model is properly detecting cars, although because of the inconsistent ground truth labeling (a lot of the far away vehicles don't have bounding boxes even if they are vehicles), the model sometimes predicts objects even when they aren't technically present in the ground truth labeling.

### Quantitative Metrics

![image](./images/graph.png)

### Analysis so far...

As seen in the image from our test set, this iteration of our model already does a very good job of detecting vehicles and constructing their proper bounding boxes. The red and green boxes align quite well, and there are no anomalies where the model is blatantly wrong. There is one minor issue where more vehicles than labeled are being detected, but this is more due to the dataset labeling not including every vehicle present in the scene.

Via the graph above, it is also evident that we have surpassed our original benchmark of an mAP around 20-30, and are sitting in the 40-50 range. This means that our model is doing better than originally anticipated on identifying proper bounding boxes.

### Next Steps

One of our next steps will be to use a Variational Autoencoder (VAE) to compress our input into a problem of a more manageable dimensionality. We also want to consider various different object detection models with various different data augmentation methods and compare how that affects the mAP.

## References

[1] S. S. A. Zaidi, M. S. Ansari, A. Aslam, N. Kanwal, M. Asghar, and B. Lee, “A Survey of Modern Deep Learning based Object Detection Models.” arXiv, May 12, 2021. doi: 10.48550/arXiv.2104.11892.

[2] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.” arXiv, Jul. 06, 2022. doi: 10.48550/arXiv.2207.02696.

[3] Y. Chen and Z. Li, “An Effective Approach of Vehicle Detection Using Deep Learning,” Comput Intell Neurosci, vol. 2022, p. 2019257, Jul. 2022, doi: 10.1155/2022/2019257.

## Contributions

### Proposal Chart

| **Member** | **Contributions**                           |
| ---------- | ------------------------------------------- |
| Ankith     | Website, writing problem definition/methods |
| Emanuel    | Wrote Introduction and Introduced Idea      |
| Jeet       | Research                                    |
| Charles    | Research                                    |
| Vikranth   | Video Presentation                          |

### Gant Chart (See Excel File in Repo For More Information)

| **Member** | **Contributions**                                         |
| ---------- | --------------------------------------------------------- |
| Ankith     | Incorporating Neural Networks, Work on Research           |
| Emanuel    | Incorporating Neural Networks and data through R-CNN, etc |
| Jeet       | Visualizations and Organization of Data                   |
| Charles    | Writing Report and Organizing Information                 |
| Vikranth   | Analyzing Dataset through Augementation/Research          |

## Repo Structure

- yolov5: Contains all the code from the ultralytics yolov5 implementation
- requirements.txt: pip installable requirements
- test.ipynb: Where all the code for finetuning YOLOv5 is on our specific dataset
- modified_yolov5s.yaml: YOLOv5 model architecture but with a modified head to predict for one class
