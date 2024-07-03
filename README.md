# ML4641Project

## Introduction/Background

For our project we wish to apply object-detection to the field of self-driving, specifically by implementing machine learning models that can detect vehicles, specifically cars, as shown in our [dataset](https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset). We indicated that we were going to use a different dataset in the project proposal, but we ended up changing to this dataset. This dataset includes around 500 training images as well as around 100 testing images, where images come with a corresponding set of labels which indicate the size, position, and type of objects located in the image. If we end up needing more data, particularly to train some of the larger deep learning models we plan on using other datasets like the [car tracking and object detection dataset](https://www.kaggle.com/datasets/trainingdatapro/cars-video-object-tracking), which contains similar images, albeit at a different angle, perhaps allowing our models to better generalize. We currently plan on finetuning a variety of state-of the art object detection models [1], such as YOLO-v7 [2], and have looked at how others have applied such models onto the problem of vehicle detection [3].

## Problem Definition

With self-driving cars becoming more and more relevant in current society, it is crucially important that they are able to detect other vehicles to prevent collisions in a reliable, quick manner. We seek to explore and expand upon existing methods of using machine learning techniques to detect vehicles.

## Methods

Due to the relatively low amount of data, we plan on increasing the amount and diversity of the data using data augmentation techniques such as color jittering, gaussian noise, or blurring. We will also normalize our data, changing the intensity of the pixels to a more consistent distribution. Finally we will resize all the images to a common size, to ensure images have a consistent input size.

Finally, for our supervised techniques, we plan on training a plain neural network on the object-detection dataset, just to gain a minimum baseline on which more advanced models can improve on. We then plan on fine-tuning state of the art object detection models, such as YOLO or R-CNN. Finally, we will look at more novel object detection methods, such as those involving ViTs (Vision Transformers). As these models require lots of computational power to train, we plan on training them via either google colab or PACE-ICE.

### Implemented Method

We currently have fine-tuned a small YOLO-v5 model on our dataset, with all the code present to do so in test.ipynb. We used Batch Gradient Descent over 20 epochs, using the Adam optimizer with a learning rate of .0001.

## Results and Discussion

We plan to use accuracy, latency, as well as mAP (mean average precision), a benchmark often used for detection problems. While it may be unrealistic to achieve in the tight timeframe of this project, we hope to be able to create/fine-tune object detection models capable of detecting vehicles with quantitative metrics at the level other papers have achieved on the same problem. For example, [3] was able to train a YOLOv3 model to achieve a mAP of 72.8. Realistically though, we hope to be able to train/finetune an object-detection model at a minimum mAP of around 20-30, a base value which should hopefully indicate that our model

### Visualizations

<img src="https://github.com/Duckierstone42/ML4641Project/assets/133888360/2f584184-7b0c-42ac-a7f5-93eff9030090.png" width="500" height="500">

The above image is an example image from the training data set. The borders around each object are called the "bounding boxes", and we aim to train our model to construct boxes just like these.

<img src="https://github.com/Duckierstone42/ML4641Project/assets/133888360/18645006-08c7-4c16-92da-2bce7a2ef88b.png" width="500" height="500">

This is an image that our model has been tested on, with bounding boxes constructed for each object detected, as well as their associated confidence levels (between 0 and 1).

The image from the test set shows clear resemblance to the image from the training set, constructing accurate green bounding boxes around each car and doing so with high confidence. There are some cars in the image that do not have bounding boxes around them, meaning either the confidence level to construct one is not above some specified threshold, or they have not been identified altogether. While already quite accurate, we hope to fine-tune the model some more to identify even more objects with higher consistency.

### Quantitative Metrics

We are currently only looking at the mAP (mean average precision), which is a metric that combines values from various other metrics (confusion matrix, recall, precision, IoU). We plan on looking at other metrics by the end of the project as well, and also look at the mAP for the train dataset, not just the test dataset.

### Analysis so far...

As you can see from the visualization above, our loss steadily increases and our test mAP sharply increases in the beginning and then stops improving.

### Next Steps

We currently have fine-tuned one small version of YOLO-v5, and in the future we plan to finetune models of even different sizes on the dataset to see how the number of parameters can influence how the mAP and loss evolve. Considering how well our model performed on our dataset, we will likely look at even smaller models, to see how reducing complexity affects our accuracy. We will also look into non-linear PCA using auto-encoders, particularly variational auto-encoders, to perform dimensionality reduction and potentially visualize our dataset.

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

## Modifications

-Make sure to use the Vehicle_Detection_Image_Dataset, and also have the yolov5s.pt weights downloaded before running the script
