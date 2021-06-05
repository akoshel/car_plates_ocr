
import os

import pandas as pd
import numpy as np
import tqdm
from torchvision import transforms
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import load_json, decode, four_point_transform
from ocr_helpers import OCRDataset, Resize, CRNN, get_vocab_from_marks, collate_fn_ocr


def train_ocr():
    DATA_PATH = '../data/'
    TRAIN_SIZE = 0.9
    BATCH_SIZE_OCR = 256
    OCR_MODEL_PATH = 'ocr.pt'
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    all_marks = load_json(os.path.join(DATA_PATH, 'train.json'))
    test_start = int(TRAIN_SIZE * len(all_marks))
    train_marks = all_marks[:test_start]
    val_marks = all_marks[test_start:]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    my_ocr_transforms = transforms.Compose([
        Resize(size=(320, 64)),
        transforms.ToTensor()
    ])

    char_to_idx, idx_to_char, alphabet = get_vocab_from_marks(train_marks)

    train_ocr_dataset = OCRDataset(
        marks=train_marks,
        img_folder=DATA_PATH,
        alphabet=alphabet,
        transforms=my_ocr_transforms
    )
    val_ocr_dataset = OCRDataset(
        marks=val_marks,
        img_folder=DATA_PATH,
        alphabet=alphabet,
        transforms=my_ocr_transforms
    )
    train_ocr_loader = DataLoader(
        train_ocr_dataset,
        batch_size=BATCH_SIZE_OCR,
        drop_last=True,
        num_workers=0,  # Почему-то у меня виснет DataLoader, если запустить несколько потоков
        collate_fn=collate_fn_ocr,
        timeout=0,
        shuffle=True  # Чтобы повернутые дубли картинок не шли подряд
    )

    val_ocr_loader = DataLoader(
        val_ocr_dataset,
        batch_size=BATCH_SIZE_OCR,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn_ocr,
        timeout=0,
    )
    crnn = CRNN()
    # crnn.load_state_dict(torch.load(OCR_MODEL_PATH))
    crnn.to(device)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=3e-4, amsgrad=True, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    crnn.train()
    for epoch in range(10):
        epoch_losses = []
        print_loss = []

        for i, batch in enumerate(tqdm.tqdm(train_ocr_loader, total=len(train_ocr_loader), leave=False, position=0)):
            images = batch["image"].to(device)
            seqs_gt = batch["seq"]
            seq_lens_gt = batch["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = F.log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = F.ctc_loss(
                log_probs=log_probs,  # (T, N, C)
                targets=seqs_gt,  # N, S or sum(target_lengths)
                input_lengths=seq_lens_pred,  # N
                target_lengths=seq_lens_gt  # N
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_loss.append(loss.item())
            if (i + 1) % 20 == 0:
                mean_loss = np.mean(print_loss)
                print(f'Loss: {mean_loss:.7f}')
                scheduler.step(mean_loss)
                print_loss = []

            epoch_losses.append(loss.item())

        print(i, np.mean(epoch_losses))

        torch.save(crnn.state_dict(), OCR_MODEL_PATH)

        test_marks = load_json(os.path.join(DATA_PATH, 'test.json'))
        crnn.eval()
        resizer = Resize()

        file_name_result = []
        plates_string_result = []

        for item in tqdm.tqdm(test_marks, leave=False, position=0):

            img_path = item["file"]
            img = cv2.imread(img_path)

            results_to_sort = []
            for box in item['nums']:
                x_min, y_min, x_max, y_max = box['bbox']
                img_bbox = resizer(img[y_min:y_max, x_min:x_max])
                img_bbox = my_transforms(img_bbox)
                img_bbox = img_bbox.unsqueeze(0)

                points = np.clip(np.array(box['box']), 0, None)
                img_polygon = resizer(four_point_transform(img, points))
                img_polygon = my_transforms(img_polygon)
                img_polygon = img_polygon.unsqueeze(0)

                preds_bbox = crnn(img_bbox.to(device)).cpu().detach()
                preds_poly = crnn(img_polygon.to(device)).cpu().detach()

                preds = preds_poly + preds_bbox
                num_text = decode(preds, alphabet)[0]

                results_to_sort.append((x_min, num_text))

            results = sorted(results_to_sort, key=lambda x: x[0])
            num_list = [x[1] for x in results]

            plates_string = ' '.join(num_list)
            file_name = img_path[img_path.find('test/'):]

            file_name_result.append(file_name)
            plates_string_result.append(plates_string)

        df_submit = pd.DataFrame({'file_name': file_name_result, 'plates_string': plates_string_result})
        df_submit.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train_ocr()
