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
This is the main task. Normally, a dataset has at least 3 folders
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
