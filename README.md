# Clothing-Detection

## Data Pre-processing
### Downloading the data

The data used is from DeepFashion2 and can be found and downloaded from [here](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok)

Additionally, the data can be downloaded using [gdown](https://pypi.org/project/gdown/)

The data extracted from the zips should be placed into a file called "Original-Data"


 
The training set can be downloaded using:

    gdown https://drive.google.com/uc?id=1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK

And the validation set can be downloaded using:

    gdown https://drive.google.com/uc?id=1O45YqhREBOoLudjA06HcTehcEebR0o9y

These can then be extracted into "Original-Data" using:

    unzip Original-Data/train.zip -d Original-Data/ && unzip Original-Data/validation.zip -d Original-Data/

Or your method of choice.

Resulting in a file directory of the following:

    Original-Data
    |  train
    |  |  image
    |  |  annos
    |  validation
    |  |  image
    |  |  annos

### Convert to KITTI format
The data is converted into the KITTI format for training by first converting it into coco using [deepfashion2_to_coco.py](deepfashion2_to_coco.py) then converted from coco into KITTI using [coco2kitti.py](coco2kitti.py).

This can be done by running each file using:

    python deepfashion2_to_coco.py

and:

    python coco2kitti.py

Resulting in a file directory of the following where label contains annotations in the KITTI format:

    Original-Data
    |  train
    |  |  image
    |  |  annos
    |  |  label
    |  validation
    |  |  image
    |  |  annos
    |  |  label
    

## Training using NVIDIA transfer learning toolkit 2.0
### Using Docker
The Docker being used can be pulled by using:

    docker pull nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3

It can then be opened using:

    docker run --runtime=nvidia -it -v "$(pwd)":/workspace -p 8888:8888 nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3 

### Augmenting the data
The data is augmented using the tlt-augment feature, which uses the [augmentation spec](KITTI-augmentation.conf).

This can be run inside the docker using:

    tlt-augment -d Original-Data/train -a KITTI-augmentation.conf -o KITTI-dataset-augmented/train/

and:

    tlt-augment -d Original-Data/validation -a KITTI-augmentation.conf -o KITTI-dataset-augmented/validation/

### Dataset converison
The data can be converted into TFRecords for training using the tlt-dataset-convert feature, which uses the [conversion spec](KITTI_dataset_conversion.conf).

This can be run inside the Docker using:

    tlt-dataset-convert -d KITTI_dataset_conversion.conf -o TFRecords/


### Training
The data can be trained using the tlt-train ssd feature, which uses the [training spec](training.conf).

This can be run inside the Docker using:

    tlt-train ssd --gpus 1 -e training.conf -r /workspace/output -k nvidia-tlt

