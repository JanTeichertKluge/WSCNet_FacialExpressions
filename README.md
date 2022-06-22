# WSCNet_EmotionDetector
Usage of opencv-python to detect human faces in frames, classification of faces with pytorch using WSCNet framework.

WSCNet: Weakly Supervised Coupled Networks for Visual Sentiment Classification and Detection
By Dongyu She, Jufeng Yang, Ming-Ming Cheng, Yu-Kun Lai, Paul L. Rosin and Liang Wang
https://github.com/sherleens/WSCNet

### Introduction

We use the WSCNet framework for Visual Sentiment Classification and Detection to classify human facial expressions.
We changed the following according to the original WSCNet-Script file: 
  1) Changed Global Average Pooling to Global Max Pooling in Detection Branch
  2) Added LogSoftmax activation function to Detections- and Classifications Branch Outputs


### Requirements

See requirements.txt

### Dataset for training

We used FER2013 dataset for train the WSCNet to classify human facial expressions. The WSCNet was build on pretrained ResNet101.
We achieved an accuracy of about 59% after 20 epochs.

### Our trained models

Find the state_dict of pretrained WSCNet on FER2013 on Google Drive:
https://drive.google.com/drive/folders/1FABmh9eX2d4b_NLbHe5Hlbyem_7_ydYQ?usp=sharing
