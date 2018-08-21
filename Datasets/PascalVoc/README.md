
### Directory

#### SegmentationClass
The directory from Pascal VOC dataset containing the semantic masks.<br/>
<img src="example_images/2007_000129.png" width="100">

#### image
The directory from Pascal VOC dataset containing the raw RGB images.<br/>
<img src="example_images/2007_000129.jpg" width="100">

#### label
Turn the SegmentationClass into channel 1 range from [0,20].<br/>
<img src="example_images/2007_000129_0_20.png" width="100">

#### colabel
Extract image pairs from label and if they have same objects saving their co-segment mask.<br/>
<img src="example_images/2007_000129_2007_000323.png" width="100">

### Script

#### convert_val_anno.py
Convert channel 3 masks in SegmentationClass directory into channel 1 range [0,21] label and save them in label directory.

#### sperate_train_val.py
Sperate the labels into train and val according to the train.txt and val.txt

#### create_coseg_dataset.py
Put this script into a folder contains semantic label(e.g. label/train/).
Extract all image pairs in current folder and create co-segmentation label saving it to colabel/train/
