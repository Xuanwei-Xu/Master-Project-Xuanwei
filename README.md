## Master Project: Brain Tumor MRI Image Classification Using Optimizied ResNet-50

### Author Information
Xuanwei Xu, University of Manchester, The Faculty of Science And Engineering, Advanced Computer Science

**Supervisor Name**: Fumie Costen

### Abstract
A brain tumor defined as the uncontrolled growth of abnormal cells in the brain, which is one of the most dangerous and deadly diseases. Early detection and diagnosis will be directly related to the cure rate of the disease and the survival rate of patients. A diagnosis of the type of brain tumor is an essential step before intervention and treatment. At present, some automated artificial intelligence systems have been applied in the field of medical diagnosis. Experts have greatly improved the accuracy and efficiency of brain tumor diagnosis by using deep learning models instead of traditional diagnosis methods. However, developing a highly accurate and reliable deep learning model for brain tumor classification remains a challenge. This paper aims to propose the optimized ResNet-50 model used for classifying brain tumors given brain MRIs by introducing the concepts of contrastive learning and attention mechanisms. Specifically, the contrastive framework MoCo v2 is used to pre-train the ResNet-50 model that has added a SE block outside the structure of each residual block in its architecture, then it will be used as the encoder for supervised training and applied in the brain tumor MRI classification, which enables the model to learn and extract more useful features and relationships from data to improve the classification ability of different types of brain tumors. More importantly, this project explores and evaluates the contribution or impact of applying these two optimization methods on the overall performance of the ResNet-50 model. The classification performance of the optimized models is not only compared with the ResNet-50 model before optimization but also compared with the other three well-known CNN models: AlexNet, EfficientNetV2, MobileNetV2. All the models were previously trained on the ImageNet dataset for transfer learning except the optimized model; these pre-trained models are used for the supervised brain tumor classification task in this project by training, fine-tuning, and testing their performances on the same dataset. The results show that using MoCo v2 for self-supervised contrastive pre-training could outperform ImageNet-supervised pre-training or initialization; the classification accuracy of the contrastive pre-trained ResNet-50 model is further improved after the addition of the attention blocks and reaches 98.3% on the test set, which is higher than all other models. The experiment proved that the optimized model surpasses all these pre-trained models and the optimization of the model is effective and successful.

### Dataset
Links to the pubicly available datasets used in this project are provided below:

(https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

### Environment
**Deep Learning framework**: Pytorch 2.0.1
**Programming Language**: Python 3.7
**Operating System**: Ubuntu 16.04
**IDE**: Pycharm community edition
**Platform**: Anaconda3 2021.11(64 bits)
**Basic Steps**: Install zip Pytorch 2.0.1 from website, then extract it and place the folder into the path : "Anaconda3\envs\pytorch2.0.1\python.exe", then open this project using Pycharm and add a python interpreter using this path.

### Dirctory Description in this project

**1.** **"brain_dataset"**: This directory represents the original brain tumor dataset.

**2.** **"brain_dataset_24b"**: This directory represents the brain tumor dataset converted to RGB data format.

**3.** **"brain_dataset_24b_split"**: This directory represents the brain tumor dataset with RGB formats after splitting, of which 80% is used for the training set and 20% for the test set.

**4.** **"alexnet**, **"efficientnetV2"**, **"mobilenet"**, **"resnet50"**: These directories represents the implementations of four ImageNet-supervised pre-trained models

**5.** **"pycontrast"**: directory is a cloned repository from GitHub used to perform MoCo V2 pre-training on the ResNet-50 architecture in my project by using the brain tumor dataset. This publicly available repository was released on GitHub by Yonglong and Sun et al.; it contains Pytorch implementations of a set of well-known contrastive learning frameworks that currently have the best performance in unsupervised representation learning (State-of-the-art methods), including the MoCo series, CMC, InstDis, and so on. Therefore, these off-the-shelf frameworks can be used in this project to pre-train a specified model on a dataset to learn feature representations that could apply to downstream tasks. The link to this repository is as follows:

(https://github.com/HobbitLong/PyContrast/tree/master/pycontrast)

The repository predefines a set of training commands with respect to each framework and allows customization of the model. The default model used for pre-training in this repository is the ResNet series model, and the "--arch" option is used in the command to specify a different architecture. The architecture of ResNet-50 is built and defined in the path "pycontrast\networks\resnet.py". The command for using MoCo V2 method to pre-train ResNet-50 model on the brain tumor dataset in this project are as follows:

{
python main_contrast.py \
   --method MoCov2 \
   --cosine \
   --data_folder ../brain_dataset_24b_split \  
--multiprocessing-distributed --world-size 1 --rank 0 \
}

The weights learned by the Resnet-50 model on the dataset after using MoCo V2 pre-training are saved in the path "pycontrast\save\last_model.pth". The file with .pth extension saves the state of a pytorch model using a state dictionary, it can be loaded in models as initialized parameters. Then these MoCo V2 pre-trained weights are loaded in the ResNet-50 architecture as initialized parameters, then trained and fine-tuned on the dataset to classify brain tumors in my project. 

**6.** **"resnet50_cl"**: represents the ResNet-50 model optimized with MoCo V2 pre-training (Contrastive Learning), which means the ResNet-50 model using the pre-trained weights learned from MoCo V2 pre-training as initialization parameters(The pre-trained weights from "pycontrast" module)

**7.** **"resnet50_se"**: represents the ResNet-50 that has added an SE block after the structure of each residual block in its architecture. Then it uses ImageNet pre-trained weights as initialization parameters and applies to this brain tumor classification task by training and fine-tuning on the dataset.
For the code implementation of the SE blocks, the project uses a publicly available Github codebase named **"External-Attention-pytorch"**. This repository is published by a student from Xiamen University and integrates pytorch implementations of a large number of different attention mechanisms in object detection algorithms, such as CBAM, BAM, SE-Net, ECA-Net, SA-Net, and other well-known attention networks. The purpose of this codebase is to facilitate researchers to use of these components directly in their own research and avoid building wheels repeatedly. The link to this repository is as follows: 

(https://github.com/xmu-xiaoma666/External-Attention-pytorch/tree/master)

Similarly, the Git repository was cloned into one of the directories of this local brain tumor classification project. The architecture of the SE Attention module is built and defined in the path "resne50_se/model_file/attention/SEAttention" while the architecture of ResNet-50 is in the path "resne50_se/runs/model.py". Then the ResNet-50 model with the addition of the attention module uses ImageNet pre-trained weights as initialization and is used to classify brain tumors after training and fine-tuning on the dataset. 

**8.** **"se-pycontrast"**: represents the cloned **"pycontrast"** contains a module **"External-Attention-pytorch"**, which is a combination of these two Git repositories and used in my project for the Pytorch implementations of pre-training ResNet-50 model that has added SE blocks. The ResNet-50 architecture with the addition of SE blocks is defined in "se-pycontrast/networks/resnet.py".Similarly, the weights learned by the Resnet-50 model on the dataset after using MoCo V2 pre-training are saved in the path "se-pycontrast\save\last_model.pth". Then these pre-trained weights are loaded in the ResNet-50-SE architecture as initialized parameters, trained, and fine-tuned on the dataset used for the brain tumor classification task. The command for using MoCo V2 method to pre-train ResNet-50-se model on the brain tumor dataset in this project are as follows, it is the exactly same as the **"pycontrast"** module mentioned above because the ResNet-50 model with the addition of SE blocks is defined in "se-pycontrast/networks/resnet.py" :

{
python main_contrast.py \
   --method MoCov2 \
   --cosine \
   --data_folder ../brain_dataset_24b_split \  
--multiprocessing-distributed --world-size 1 --rank 0 \
}

**9.** **"resnet50_se_cl"**: In this research, ResNet-50-SE-CL indicates the ResNet-50-SE model optimized with MoCo V2 pre-training, which is the final optimized model. Adding the SE module to the ResNet-50 architecture is defined in the path "resnet50-se-cl/model.py". Then, using the MoCo V2 framework to pre-train it on the dataset, its pre-trained weights are used as initialization parameters to classify brain tumors after training and fine-tuning on this dataset.

**10.** **"vit"**, **"swin_transformer"**: represents two extension algorithms( they are popular transformer models), they won't mention in my dissertation.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

### Files Descriptions
The structure of all models (ImageNet pre-trained and optimized models) has the following files: **"inference.py"**, **"model.py"**, **"multi_ inference.py"**, **"plot_confusion_matrics.py"**, **"test.py"**, **"train _xx model.py"**, **"train.log"**, **"test.log**

**1.** **model.py**: The architecture and components of the model are defined as well as the necessary neural network modules.

**2.** **"train _xx model.py"**: Various transformations for the data in the training and test sets are defined, as well as various parameter settings, and the training process and prediction process of the model, and the evaluation process.

**3.** **test.py**: Defines the process of calculating model performance metrics, measuring the gap between model predictions and actual labels used for model validation and evaluation.

**4.** **"plot_confusion_matrics.py"**： Defines the process of drawing a multi-class confusion matrix for the well-trained model. A confusion matrix is used for each model to visualize its distribution of prediction results on the same dataset, which counts the number of correctly and incorrectly classified samples for each type of brain tumor. This four-class confusion matrix can be converted into a one-vs-all class confusion matrix (binary-class confusion matrix) to calculate class-wise performance metrics. Take one of the brain tumor classes, glioma, as an example. In the form of a binary confusion matrix, the positive class refers to the type of gliomas, and the negative class represents "NOT gliomas" (Meningioma, Pituitary, and No-tumor). In this way, the metrics of an individual model to classify each type of brain tumor can be measured by using corresponding formulas; these metrics have an important reference value for evaluating the performance of the model.

**5.** **"inference.py"**: This file is used to classify brain tumors given a single input MRI image using the model，the input image path can be modified to change a different MRI image. Click the "main" function in this file below to run a model's classification prediction of one MRI image. There are four MRI images with respect to four types of brain tumors (glioma, meningioma, no-tumor, and pituitary) at the bottom of the "brain-tumor-master" directory, they are selected from our dataset and used as samples for each model's prediction. These files with jpg format and named "0, 1, 2, 3" respectively.

**6.** **"multi_ inference.py"**: Used to classify multiple input MRI images at one time, it can accept multiple inputs for classification at one time compared to the "inference.py" file. Click the "main" function in this file below to run a model's classification prediction of multiple MRI images at one time. There are four MRI images with respect to four types of brain tumors (glioma, meningioma, no-tumor, and pituitary) at the bottom of the "brain-tumor-master" directory, they are selected from our dataset and used as samples for each model's prediction. These files with jpg format and named "0, 1, 2, 3" respectively.

**7.** **"train.log"**：The model is trained with metrics tracking and recording, loss and accuracy.

**8.** **"test.log"**: Four class-wise performance metrics (accuracy, precision, specificity, sensitivity) computed by the well-trained model on the test set were recorded.
