---
layout: post
title:  "Train and test py-Faster-RCNN on a new dataset"
date:   2017-05-04 10:37:00
---

In post shares the steps to build neural networks using FRCNN on our dataset.

# Installing
Install py-Faster-RCNN using its [original github](https://github.com/rbgirshick/py-faster-rcnn) or follow some tutorials, such as [this one](https://huangying-zhan.github.io/2016/09/22/detection-faster-rcnn.html). Personally, I found the tutorial is easier to follow.
Note: this package requires OpenCV

# Setting up the dataset
This is the main task.
## Dataset structure
Normally, a dataset has at least 3 folders
```
|--dataset_folder
       |--Annotations
            |--*.txt/xml
       |--Images
            |--*.png/jpg       
       |--ImageSets
            |--train/test/val/trainval.txt
```
Number of files and filenames in `Images` and `Annotations` are identiccal (except file extensions). Files in `ImageSets` will contain filename only (discarding file extension), simply thinking this is just a file list.
There are 2 choices for files in `Annotations`: `txt` or `xml`. For now, follow `xml` version to match with `VOC2007`.
A sample annotation file is (the annotation filename: `n02802426_999.xml`):
```
<annotation>
        <folder>n02802426</folder>
        <filename>n02802426_999</filename>
        <source>
                <database>ImageNet database</database>
        </source>
        <size>
                <width>500</width>
                <height>500</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>n02802426</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>167</xmin>
                        <ymin>2</ymin>
                        <xmax>450</xmax>
                        <ymax>233</ymax>
                </bndbox>
        </object>
</annotation>
```
## Preparing dataset files
New python files are required to access to the new dataset.
1. Add `$FRCN/lib/datasets/[yourDS].py`: reading/loading a part of the whole dataset. Basically, we can clone an existing file, e.g. [`pascal_voc.py`](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py), then modify it to fit the new dataset (find and replace), e.g. [`inria.py`](https://github.com/deboc/py-faster-rcnn/blob/master/lib/datasets/inria.py) or [basketball.py](https://github.com/Huangying-Zhan/py-faster-rcnn/blob/master/lib/datasets/basketball.py)
    1. Modify `self._classes` in the constructor function to fit your dataset.
    2. Be careful with the *extensions* of your image files. See `image_path_from_index` in `inria.py`.
    3. Write the function for parsing annotations. See `_load_inria_annotation` in `inria.py`.
    4. Do not forget to add `import` syntaxes in your own python file and other python files in the same directory.
2. Add `$FRCN/lib/datasets/[yourDS]_eval.py`: contain evaluating function, e.g. mAP
3. Update `$FRCN/lib/factory.py`: loading all sets of the dataset. Manually assign data sets and *devkit_path*
4. Update `$FRCN/lib/datasets/imdb.py` if needed, e.g. ImageNet images start with index 0 in row and col while PASCAL VOC dataset starts with index 1, we will update this in the function `append_flipped_images()`    
5. Adding a config file `$FRCN/experiments/cfgs/config.yml` (full configurable keywords can be found [here](https://github.com/rbgirshick/py-faster-rcnn/blob/96dc9f1dea3087474d6da5a98879072901ee9bf9/lib/fast_rcnn/config.py)). We can directly modify the `$FRCN/experiments/cfgs/faster_rcnn_end2end.yml`, e.g. setting `EXP_DIR` first and others if necessary, or it's OK to create a new config file and modify the training script to point to the new config file. The file location can be specific in the calling command later.

## Prepare network and pre-trained model
Basically we don't need to train the model from scratch unless you have a huge dataset which is comparable to ImageNet. Because a pre-trained Faster R-CNN contains a lot of good lower level features, which can be used generally.
FRCNN provides ZF and VGG pre-trained VGG16. For example, to load the pre-trained ZF network
```
$ cd $FRCN/models
# copy a well-defined network and make modification based on it
$ mkdir your_project
$ cp ./pascal_voc/ZF/faster_rcnn_end2end/* ./your_project/
$ cd your_project
```
The content of `models/pascal_voc/ZF/faster_rcnn_end2end` is: `solver.prototxt` (tells the program where to find your ConvNet structure prototxt and some training setups, such as learning rate, learning policy, etc.), `train.prototxt` (describes the network structure, including number of layer, type of layer, number of neurons in each layer, etc.), and `test.prototxt`
Some suggested modifications:
  * `solver.prototxt`
    * train_net
    * snapshot_prefix
  * `train.prototxt` & `test.prototxt`: we need to update the number of output in final layers. For example, in a binary classification dataset, we only need 2 classes (background + basketball) and 8 output for bounding box regressor. Orignial pascal_voc have 21 classes including background and 21*4 bounding box regressor output.
```
$ cd $FRCN/models/basketball
$ grep 21 *
$ grep 84 *
# These two commands help you to check the lines that you should modify in the files.
```  
    There are two more items we need to modify. Since we are fine-tuning a pre-trained ConvNet model on our own dataset and the number of output at last fully-connected layers (clsscore & bboxpred) has been changed, the original weight in pre-trained ConvNet model is not suitable for our current network. The dimension is totally different. The details can be refered to Caffe's fine-tuning tutorial. The solution is to rename the layers such that the weights for the layers will be initialized randomly instead of copying from pre-trained model (actually copying from pre-trained model will cause error).
    ```
    name: "cls_score" -> name: "cls_score_basketball"
    name: "bbox_pred" -> name: "bbox_pred_basketball"
    ```
    However, renaming the layers may cause problems in later parts since "clsscore" and "bboxpred" are used as keys in testing. Therefore, in the training part, we can train the model accroding to the following procedure.
      1. Rename the layers to `cls_score_basketball` and `bbox_pred_basketball`
      2. Fine-tune pre-trained Faster R-CNN (FRCN) model and snapshot at iteration 0. Let's call the snapshot `Basketball_0.caffemodel`. Stop training.
      3. Rename the layers back to `cls_score` and `bbox_pred`.
      4. Fine-tune `Basketball_0.caffemodel` to get our final model.
## Training and Evaluation
Before training on your new dataset, you may need to check $FRCN/data/cache to remove caches if necessary. Caches stores information of previously trained dataset. It may cause problem while training.
### Training
1. Rename the layers (to avoid override the pre-trained network's architecture)
  As mentions in the previous part, rename the two layers.
  Reminder: if you are using find and replace, please find the name with quotes(i.e. "cls_score"). If you just search for `cls_score`, without quotes, it may also replace some other layers since there is a layer named rpn_cls_score.
2. First fine-tuning
  The purpose of first fine-tuning is to get a caffemodel which has two outputs at final fully-connected layers.
  ```
  $ ./tools/train_net.py --gpu 0 --weights data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel --imdb basketball_train --cfg experiments/cfgs/config.yml --solver models/basketball/solver.prototxt --iter 0
  ```
  In general, what we do in the previous command is: call the `train_net` script/procedure using GPU. Network's initial weights are loaded from `data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel`, code to deal with dataset is `basketball_train` and configuration file is located at `experiments/cfgs/config.yml`. Finally, the solver information is loaded from `models/basketball/solver.prototxt`. 
  After this fine-tuning, we should get the model we needed.
3. Rename the layers back (to make it consisent with the test set)
  Rename the two layers back to "cls_score" and "bbox_pred".
4. Second fine-tuning
  This fine-tuning should train models for our final use. The pre-trained model in this stage is the model we saved in stage 2.
  ```
  $ ./tools/train_net.py --gpu 0 --weights output/basketball/train/zf_faster_rcnn_basketball_iter_0.caffemodel --imdb basketball_train --cfg experiments/cfgs/config.yml --solver models/basketball/solver.prototxt --iter 10000
  ```
  Basically, it loads the network's weights we fine-tuned in the 2nd step (`iter_0` because we only train 1 pass), this time, we train for a long time, e.g. 10k iterations
### Evaluating
To test the performance of trained model, we can use the provided test_net.py for the purpose.
```
$ ./tools/test_net.py --gpu 0 --def models/basketball/test.prototxt --net output/basketball/train/zf_faster_rcnn_basketball_iter_20000.caffemodel --imdb basketball_val --cfg experiments/cfgs/config.yml
```
Why we only use 10k iteration for training but testing load a model from 20k iterations?
