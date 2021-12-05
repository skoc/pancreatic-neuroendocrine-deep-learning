# pancreatic-neuroendocrine-deep-learning
Pancreatic Neuroendocrine Tumors Image Analysis Using Deep Learning

Notebooks
-   
#### preprocessing

* Input: QuPath tiled images
* Operation: Clean tiles according to non-square and min(class sample) per class
* Output: Folder created with cleaned and equal sized tiles per class

#### histolab-preprocess-pannet

* Input: WSI
* Operation: Random or Grid tiling WSI
* Output: Folder Created with WSI's tiles

#### train-finetune

* Input: Cleaned Tiles and Trained Model Weights - ciga
* Operation: Training
* Output: Folder Created with trained models

#### test

* Input: Trained model and Grid tiles of WSI
* Operation: Prediction
* Output: CSV Created with tile,x,y,prediction,confidence values

#### visuzalization

* Input: Scaled WSI and Prediction CSV of WSI
* Operation: Place predicted tiles on scaled WSI
* Output: Image Created with predictions on WSI

#### visualize-class distribution

* Input: Prediction CSV of WSI
* Operation: Plot per class prediction distribution
* Output: Image Created with class predictions individually

#### umap-ciga

* Input: Extracted features from the pretrained model as csv - ciga
* Operation: UMAP - dimensionality reduction
* Output: Interactive UMAP with class distribution (open in jupyter notebook not lab)
