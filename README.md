# Extracting_Urban_Features_SSL

This work aims to study the urban features that affect the crime rate of an area. Such insights would help in better planning of urban infrastructure in the future to tackle various global issues. In our study, we treat the extraction of such urban features from satellite images as an object detection and classification problem. Given the advancements in the field of self-supervised learning (SSL) and the huge availability of unlabeled satellite data, two self-supervised models, Deformable DETRreg and Masked Auto Encoder were tested and compared to detect urban features in the input satellite images. The models were evaluated on two levels. Mean Average Precision (MAP) was used to evaluate the ability to detect urban features. To evaluate whether the detected urban features are relevant, the classification performance of the feature maps of satellite images from the SSL models into high and low crime rates is compared against a baseline classifier.

Procedure to run the scripts:

Swav pretraining:

Deformable DETR pretraining:

Deformable DETR finetuning:

Deformable DETR prediction:

MAE pretraining:

MAE finetuning:

MAE prediction:
