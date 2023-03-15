import os
import argparse

# file structure is
# ./
# --dataset/
# ----------archive/
# ----------License Plates.v3-original-license-plates.voc/

parser = argparse.ArgumentParser(description="Dataset Organizer")
# parser.add_argument("--dataset_path", type=str, default=os.getenv("data_path"))
parser.add_argument("--dataset_path", type=str, default="./dataset")
args = parser.parse_args()


def process_kaggle_plate_dataset(dataset_path, folder_name, train_list):
    # process the dataset downloaded from https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download
    data_path = os.path.join(dataset_path, folder_name)
    img_list = os.listdir(os.path.join(data_path, "images"))
    for img_name in img_list:
        img_path = os.path.join(data_path, "images", img_name)
        annotation_path = os.path.join(data_path, "annotations", img_name.replace(".png", ".xml"))
        file_info = img_path + ", " + annotation_path + "\n"
        train_list.append(file_info)
    return train_list


def process_split_roboflow_plate_dataset(data_path, file_list):
    data_list = os.listdir(data_path)
    img_list = [filename for filename in data_list if ".jpg" in filename]
    for img_name in img_list:
        img_path = os.path.join(data_path, img_name)
        annotation_path = os.path.join(data_path, img_name.replace(".jpg", ".xml"))
        file_info = img_path + ", " + annotation_path + "\n"
        file_list.append(file_info)
    return file_list


def process_roboflow_plate_dataset(dataset_path, folder_name, train_list, val_list, test_list):
    # process the dataset downloaded from https://public.roboflow.com/object-detection/license-plates-us-eu/3
    # data should be in the PASCAL VOC format
    data_path = os.path.join(dataset_path, folder_name)
    train_list = process_split_roboflow_plate_dataset(os.path.join(data_path, "train"), train_list)
    val_list = process_split_roboflow_plate_dataset(os.path.join(data_path, "valid"), val_list)
    test_list = process_split_roboflow_plate_dataset(os.path.join(data_path, "test"), test_list)
    return train_list, val_list, test_list


if __name__ == "__main__":
    subsets = os.listdir(args.dataset_path)
    train_list = []
    val_list = []
    test_list = []

    for subset in subsets:
        if subset == "archive":
            train_list = process_kaggle_plate_dataset(args.dataset_path, subset, train_list)
        elif subset == "License Plates.v3-original-license-plates.voc":
            train_list, val_list, test_list = process_roboflow_plate_dataset(args.dataset_path, subset, train_list,
                                                                             val_list, test_list)

    train_file = os.path.join(args.dataset_path, "train.txt")
    val_file = os.path.join(args.dataset_path, "val.txt")
    test_file = os.path.join(args.dataset_path, "test.txt")

    for entry in train_list:
        with open(train_file, 'a') as f:
            f.write(entry)
            f.close()
    for entry in val_list:
        with open(val_file, 'a') as f:
            f.write(entry)
            f.close()
    for entry in test_list:
        with open(test_file, 'a') as f:
            f.write(entry)
            f.close()
