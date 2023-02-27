# Install required libs
#!pip install -U segmentation-models-pytorch albumentations --user 

#!pip uninstall -y segmentation-models-pytorch

# Loading data
# For this example we will use CamVid dataset. It is a set of:

# train images + segmentation masks
# validation images + segmentation masks
# test images + segmentation masks
# All images have 320 pixels height and 480 pixels width. For more inforamtion about dataset visit http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/.

# Conda environment:
# conda activate elements_mapping
# import resource
# import platform
# import sys

# def memory_limit(percentage: float):
#     """
#     只在linux操作系统起作用
#     """
#     if platform.system() != "Linux":
#         print('Only works on linux!')
#         return
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#     resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))

# def get_memory():
#     with open('/proc/meminfo', 'r') as mem:
#         free_memory = 0
#         for i in mem:
#             sline = i.split()
#             if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
#                 free_memory += int(sline[1])
#     return free_memory

# def memory(percentage=0.9):
#     def decorator(function):
#         def wrapper(*args, **kwargs):
#             memory_limit(percentage)
#             try:
#                 function(*args, **kwargs)
#             except MemoryError:
#                 mem = get_memory() / 1024 /1024
#                 print('Remain: %.2f GB' % mem)
#                 sys.stderr.write('\n\nERROR: Memory Exception\n')
#                 sys.exit(1)
#         return wrapper
#     return decorator

# @memory(percentage=0.9)
# def allocate_memory():
#     print('My memory is limited to 90%.')

# allocate_memory()
####################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_DIR = 'project_data/'

# # load repo with data if it is not exists
# if not os.path.exists(DATA_DIR):
#     print('Loading data...')
#     os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
#     print('Done!')


x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_annot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_annot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_annot')


# helper function for data visualization
def visualize(image, mask):
    """PLot images in one row."""
    # n = len(images)
    n = 8
    plt.figure(figsize=(150, 25))
    # for i, (name, image) in enumerate(images.items()):

    plt.subplot(1, n, 1)
    plt.xticks([])
    plt.yticks([])
    # plt.title(' '.join(name.split('_')).title())
    plt.imshow(image)

    for j in range(7):
        plt.subplot(1, n, j + 2)
        plt.xticks([])
        plt.yticks([])
        # plt.title(' '.join(name.split('_')).title())
        # plt.imshow(image[j, :, :])
        plt.imshow(mask[j, :, :], cmap='gray', vmin=0, vmax=1)
    plt.show()

# Dataloader
# Writing helper class for data extraction, tranformation and preprocessing
# https://pytorch.org/docs/stable/data

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
    #            'tree', 'signsymbol', 'fence', 'car', image, mask = dataset[5] # get some sample

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            # classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # list file names in the self.ids list
        self.sem_ids = os.listdir(images_dir)
        self.label_ids = os.listdir(masks_dir)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.sem_ids]
        self.masks_fps = [os.path.join(masks_dir, label_id) for label_id in self.label_ids]
        
        # print("maks_fps:", self.masks_fps)

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]


        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        # print(image.shape)
        # converting the file dimension in [N, C, H, W] order
        # image = np.transpose(image, (2, 0, 1))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print('image type:', image.dtype)
        # print(self.masks_fps[i])
        mask = np.load(open(self.masks_fps[i], 'rb'))
        # mask = bytes(mask)
        # print('mask type:', mask.shape, mask.dtype)

        mask = np.transpose(mask, (1, 2, 0))

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # print(image.shape)
        # print(mask.shape)

        # plt.imshow(mask[1, :, :], cmap='gray', vmin=0, vmax=255)
        # plt.show()
        # print(mask[0, :, :].max())
        print()
        print('image:', self.images_fps[i], 'mask:', self.masks_fps[i])
        # print('mask', mask.shape, type(mask))
        return image, mask
        
    def __len__(self):
        return len(self.sem_ids)

# Lets look at data we have


# dataset = Dataset(x_train_dir, y_train_dir)

# image, mask = dataset[5] # get some sample


# visualize(
#     image=image, 
#     mask=mask.squeeze(),
# )

### Augmentations

# Data augmentation is a powerful technique to increase the amount of your data and prevent model overfitting.  
# If you not familiar with such trick read some of these articles:
#  - [The Effectiveness of Data Augmentation in Image Classification using Deep
# Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
#  - [Data Augmentation | How to use Deep Learning when you have Limited Data](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
#  - [Data Augmentation Experimentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b)

# Since our dataset is very small we will apply a large number of different augmentations:
#  - horizontal flip
#  - affine transforms
#  - perspective transforms
#  - brightness/contrast/colors manipulations
#  - image bluring and sharpening
#  - gaussian noise
#  - random crops

# All this transforms can be easily applied with [**Albumentations**](https://github.com/albu/albumentations/) - fast augmentation library.
# For detailed explanation of image transformations you can look at [kaggle salt segmentation exmaple](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) provided by [**Albumentations**](https://github.com/albu/albumentations/) authors.

import albumentations as albu

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                # albu.RandomBrightness(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.IAASharpen(p=1),
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.RandomContrast(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


#### Visualize resulted augmented images and masks

# augmented_dataset = Dataset(
#     x_train_dir, 
#     y_train_dir, 
#     augmentation=get_training_augmentation(), 
# )

# same image with different random transforms
# image, mask = augmented_dataset[5]
# visualize(image=image, mask=mask)

import torch
import numpy as np
import segmentation_models_pytorch as smp

torch.cuda.empty_cache()
# torch.cuda.set_per_process_memory_fraction(0.9, 0)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'
DEVICE = 'cpu'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS,
    classes = 7, 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn)
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.losses.DiceLoss("multilabel")

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.01),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    # verbose=True
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    # verbose=True,
)

# train model for 40 epochs

max_score = 0

#################################
n = np.random.choice(len(train_dataset))

train_dataset_vis = Dataset(
    x_train_dir, 
    y_train_dir, 
    # augmentation=get_training_augmentation(), 
    # preprocessing=get_preprocessing(preprocessing_fn)
)


image_vis = train_dataset_vis[n][0].astype('uint8')
image, gt_mask = train_dataset[n]

gt_mask = gt_mask.squeeze()

x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

pr_mask = model.predict(x_tensor)
pr_mask = (pr_mask.squeeze().cpu().numpy().round())

# print('pr_mask:', pr_mask.shape, type(pr_mask))

#################################

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

## Test best saved model

# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

## Visualize predictions

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
)

for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )