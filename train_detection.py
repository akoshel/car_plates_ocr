import os
import gc
import json
import glob
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import seaborn as sns
import tqdm


from torchvision import transforms

import PIL
from PIL import Image, ImageDraw
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_json, simplify_contour, npEncoder
from detection_helpers import get_detector_model, DetectionDataset


def train_detection():
    DATA_PATH = '../data/'
    TRAIN_SIZE = 0.9
    BATCH_SIZE = 128
    DETECTOR_MODEL_PATH = 'detector.pt'
    OCR_MODEL_PATH = 'ocr.pt'
    EPOCH_NUM = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    all_marks = load_json(os.path.join(DATA_PATH, 'train.json'))
    test_start = int(TRAIN_SIZE * len(all_marks))
    train_marks = all_marks[:test_start]
    val_marks = all_marks[test_start:]

    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = DetectionDataset(
        marks=train_marks,
        img_folder='../data/',
        transforms=my_transforms
    )
    val_dataset = DetectionDataset(
        marks=val_marks,
        img_folder='../data/',
        transforms=my_transforms
    )
    print("train dataset")
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    print("dataloader")
    torch.cuda.empty_cache()
    gc.collect()
    model = get_detector_model()
    # model.load_state_dict(torch.load(DETECTOR_MODEL_PATH))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

    model.train()
    print("train started")
    for epoch in range(EPOCH_NUM):

        print_loss = []
        for i, (images, targets) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            print_loss.append(losses.item())

            if (i + 1) % 20 == 0:
                mean_loss = np.mean(print_loss)
                print(f'Loss: {mean_loss:.7f}')
                scheduler.step(mean_loss)
                print_loss = []

    torch.save(model.state_dict(), DETECTOR_MODEL_PATH)
    THRESHOLD_SCORE = 0.93
    TRESHOLD_MASK = 0.05

    preds = []
    model.eval()
    test_images = glob.glob(os.path.join(DATA_PATH, 'test/*'))
    for file in tqdm.tqdm(test_images, position=0, leave=False):

        img = Image.open(file).convert('RGB')
        img_tensor = my_transforms(img)
        with torch.no_grad():
            predictions = model([img_tensor.to(device)])
        prediction = predictions[0]

        pred = dict()
        pred['file'] = file
        pred['nums'] = []

        for i in range(len(prediction['boxes'])):
            x_min, y_min, x_max, y_max = map(int, prediction['boxes'][i].tolist())
            label = int(prediction['labels'][i].cpu())
            score = float(prediction['scores'][i].cpu())
            mask = prediction['masks'][i][0, :, :].cpu().numpy()

            if score > THRESHOLD_SCORE:
                # В разных версиях opencv этот метод возвращает разное число параметров
                # Оставил для версии colab
                contours, _ = cv2.findContours((mask > TRESHOLD_MASK).astype(np.uint8), 1, 1)
                #             _,contours,_ = cv2.findContours((mask > TRESHOLD_MASK).astype(np.uint8), 1, 1)
                approx = simplify_contour(contours[0], n_corners=4)

                if approx is None:
                    x0, y0 = x_min, y_min
                    x1, y1 = x_max, y_min
                    x2, y2 = x_min, y_max
                    x3, y3 = x_max, y_max
                else:
                    x0, y0 = approx[0][0][0], approx[0][0][1]
                    x1, y1 = approx[1][0][0], approx[1][0][1]
                    x2, y2 = approx[2][0][0], approx[2][0][1]
                    x3, y3 = approx[3][0][0], approx[3][0][1]

                points = [[x0, y0], [x2, y2], [x1, y1], [x3, y3]]

                pred['nums'].append({
                    'box': points,
                    'bbox': [x_min, y_min, x_max, y_max],
                })

        preds.append(pred)

    with open(os.path.join(DATA_PATH, 'test.json'), 'w') as json_file:
        json.dump(preds, json_file, cls=npEncoder)


if __name__ == "__main__":
    print("go")
    train_detection()
