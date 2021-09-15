# DL-Healthcare

UIUC - CS 598 - Deep Learning for Healthcare final project

See "COVID X-Ray Image Classification using Deep Learning.pdf" for more details.

# ABSTRACT 

Background and Objective: Novel Coronavirus also known as COVID-19 originated in Wuhan, China around December 2019 and has now spread across the world. The study tries to classify COVID-19 chest X-Ray images against pneumonia and healthy patients using convolutional neural networks (CNNs).

Materials and Methods: In this study, we try to develop a prediction model as suggested in the paper, FLANNEL (Focal Loss bAsed Neural Network EnsembLe) for COVID-19 detection [2], which uses several standard convolutional neural network models and fuse them together for better accuracy using Ensemble methods. These publicly available datasets have been used for preliminary analysis, Kaggle Chest X-Ray images, IEEE COVID-19 data set and ChestX-ray database. The images belong to one of the four classes: “Normal” i.e. healthy, “Bacterial Pneumonia”, “Viral Pneumonia” and “COVID-19”.  The data set comprises of images that fall into the above categories. Some cleanup was done to the datasets to remove images with obvious leakage. This resulted in 293 COVID-19 examples, 1,576 normal examples, 2,773 bacterial pneumonia examples and 1,494 viral pneumonia examples for training and evaluation (compared to 100 COVID-19 examples, 1,118 normal examples 2,787 bacterial pneumonia examples and 1,503 viral pneumonia in the original FLANNEL paper [2] ).  Additionally, resizing and scaling was done as part of preprocessing, so the images are consistent across all sources. Step 1 of the process uses 9 standard CNN models trained using the standard image transforms and enhanced image transforms (random rotation, random brightness change, and random contrast change). The ensemble model is used to better address the class imbalance challenge and the quality of images. This model fuses the learnings from the best 5 base models and combines them using an importance weight and uses a focal loss to back propagate the learnings. 

Results: The base training is done using models InceptionV3, VGG19-bn, ResNeXt101, Resnet152, Densenet161, Alexnet, ShuffleNet, MNASNet and SqueezeNet. Training was done using standard image transformations and image augmentation transformation. The best 5 models are selected based on the macro F1 score. Based on this criterion, the best performing models are Resnet152, Densenet161, Alexnet, VGG19-bn and ResNeXt101. The results are fed to the Ensemble model with focal loss and it achieved a F1 score of 0.81. For COVID-19 identification, the model achieves a macro-F1 score of 0.97, both of which are an improvement over the original FLANNEL [2] paper.

Discussion: Based on the results, most base models are not performing better with the enhanced image transforms (Table 1).  As a result, enhanced image transformation was not used in developing the final models.  The final model in this paper has improved performance over the base model evaluation in the original paper. The number of images in the dataset has increased from the time of publishing the original paper and this is one of the reasons for the better performance of base models. The number of publicly available images for COVID-19 is still very limited. Also, for this paper we have used data from 2 different data sets and the quality of X-Rays and additional markings train the model to predict COVID-19 images with much higher accuracy rate. The Ensemble step is designed to use the weights from multiple base learners to increase robustness of the classification and to tackle class imbalance issue [2]
On average we have seen around 5 mins per epoch, and this significantly slows down the process. Based on the limited resources available to students, it is time consuming to efficiently run and compare the various models.  As a result, epochs were limited from the original 200.

Conclusion:  The model proposed in this paper has improved performance over the original FLANNEL [2] paper as measured by the COVID-19 F1 and Macro-F1. The authors of this paper believe this is due to increased numbers of COVID-19 images in the dataset since the time of the original paper publication.  Also, the models chosen in this paper performed better than the models chosen in the original paper.

Keywords
COVID-19 detection, convolutional neural networks (CNNs), computer-assisted radiographic image interpretation, class imbalance, image classification, ensemble models


### Data Sets

COVID-19 Chest X-Ray Dataset. This is a public dataset containing the chest X-Rays of patients who are positive or suspected of COVID-19 or other viral and bacterial pneumonias. This data is collected from public sources. This data set has 542 images from 262 people. The data set is available at
https://github.com/ieee8023/covid-chestxray-dataset
COVID-19 X-Ray Dataset This dataset contains X-Rays of patients with COVID-19, pneumonia, and no disease. This dataset is a combination of the data from multiple sources. This contains 127 images collected from COVID-19 dataset and Normal and pneumonia images are from ChestX-ray8 database. This data set is available at https://github.com/muhammedtalo/COVID-19
Chest X-Ray Images (Pneumonia) This dataset has enough non-COVID-19 and “normal” images to compare to COVID-19 images. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

