# Implementation of YOLOv3 for License Plate Detection

## Dependencies

The code has been tested on Ubuntu 18.04 with Python 3.7, PyTorch 1.10.0, and CUDA 11.3. All required packages are
listed in `environment.yml`. You can use [Anaconda](https://www.anaconda.com/products/individual) to set up a Python
environment by running

```
conda env create -f environment.yml
```

After the installation finishes, activate the environment by

```
conda activate yolov3
```

## Datasets

Two datasets are used in this repository to train and evaluate the model. They can be found from
[here](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) and
[here](https://public.roboflow.com/object-detection/license-plates-us-eu). Note that when downloading data from
roboflow, download the Pascal VOC format. After downloading the datasets, reorganize the data according to the following
file structure

```
YOLO3_plate_recognition/
|-- environment.yml
|-- eval_plate.py
|-- loss.py
|-- model.py
|-- plate_dataset.py
|-- process_dataset.py
|-- README.md
|-- train_plate.py
|-- utils.py
|-- dataset/
|   |-- archive/
|   |-- License Plates.v3-original-license-plates.voc/
```

The datasets are split such that the Kaggle dataset and the training set of the roboflow dataset will be used for
training, while the validation set and test set of the roboflow dataset are reserved for validation and testing,
respectively. Run the following command to split the dataset.

```
python process_dataset.py
```

## Training

To train the model, use the command below

```
python train_plate.py --conf_thres 0.2 --AP_iou_thres 0.5 --nms_iou_thres 0.45 \
                      --label_iou_thres 0.5 --num_workers 1 --batch_sz 16 \
                      --scheduler_step 200 --scheduler_rate 0.1 --num_epochs 400 \
                      --log_freq 1000 --save_freq 100 --model_name yolov3_plate
```

The above command will train the model for 400 epochs. The checkpoint weights are saved in `log/yolov3_plate` every 100
epochs. E.g., the weights obtained after training the model for 400 epochs can be found in `log/yolov3_plate/400`. In
the same folder, the Tensorboard events which log the training/validation loss and sample outputs is also available. To
view these events, run

```
cd log/yolov3_plate
tensorboard --logdir=$PWD --samples_per_plugin=images=100
```

## Evaluation

After the above training is completed, the model weights are saved in `log/yolov3_plate/400`. To evaluate the model with
these pretrained weights on the test set, run

```
python eval_plate.py --model_name eval_yolov3_plate \
                     --pretrained_model log/yolov3_plate/400 --img_sz 416 \
                     --conf_thres 0.4 --AP_iou_thres 0.5 --nms_iou_thres 0.4 \
                     --label_iou_thres 0.5 --device cuda --split 2 --log_freq 1 
```

If you want to run evaluation on the validation set, change `--split 2` to `--split 1`.

A tensorboard event is saved in `log/eval_yolov3_plate` during evaluation. To visualize these results, run

```
cd log/eval_yolov3_plate
tensorboard --logdir=$PWD --samples_per_plugin=images=100
```

## Acknowledgments

The code is heavily based on
this [repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3)
.
