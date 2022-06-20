import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_yaml,
                           colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import box_iou
from utils.torch_utils import select_device, time_sync


def save_one_json(predn, jdict, path, class_map, images_name_id):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': images_name_id[image_id],
            'category_id': class_map[int(p[5])]+1,
            'bbox': [round(x, 1) for x in b],
            'score': round(p[4], 4),
            'segmentation': []})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        callbacks=Callbacks(),
        compute_loss=None,
):

    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if pt:  # check --weights are trained on --data
        ncm = model.model.nc
        assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                            f'classes). Pass correct combination of --weights and --data that are trained together.'
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad = 0.0
    rect = pt  # square inference for benchmarks
    dataloader = create_dataloader(data['test'],
                                    imgsz,
                                    batch_size,
                                    stride,
                                    pad=pad,
                                    rect=rect,
                                    workers=workers,
                                    prefix=colorstr(f'{"test"}: '),
                                    task='test')[0]

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = list(range(1000))
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map75, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    with open('/DATA/test/Test_Images_Information.json') as f:
        json_data = json.load(f)

        # generate images_id_name dict
        json_images = json_data["images"]
        images_name_id = {}

        for image in json_images:
            images_name_id[image['file_name'].split('.')[0]] = image['id']

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im, augment=True, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False, max_det=50)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            save_one_json(predn, jdict, path, class_map, images_name_id)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        callbacks.run('on_val_batch_end')

    # Save JSON
    w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    pred_json = str(save_dir / f"{w}_preds.json")  # predictions json

    LOGGER.info(f'\nsaving {pred_json}...')
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    # conf_cut
    conf_thres_try = conf_thres
    pred_json_cut = str(save_dir / f"{w}_preds_cut.json")

    # checking.. 0.01
    while True:
        print('checking conf_thres ', conf_thres_try, end=' - ')

        if os.path.exists(pred_json_cut):
            os.remove(pred_json_cut)

        jdict_cut = []
        for det in jdict:
            if det['score'] >= conf_thres_try:
                jdict_cut.append(det)
        
        with open(pred_json_cut, 'w') as f:
            json.dump(jdict_cut, f)
        
        json_size = os.path.getsize(pred_json_cut) / (1000.0 * 1000.0)
        print('json_size ', json_size)
        if json_size < 20:
            break
        
        conf_thres_try = round(conf_thres_try + 0.01, 3)
    
    conf_thres_try = round(conf_thres_try - 0.01, 3)

    # checking.. 0.001
    while True:
        print('checking conf_thres ', conf_thres_try, end=' - ')

        if os.path.exists(pred_json_cut):
            os.remove(pred_json_cut)

        jdict_cut = []
        for det in jdict:
            if det['score'] >= conf_thres_try:
                jdict_cut.append(det)
        
        with open(pred_json_cut, 'w') as f:
            json.dump(jdict_cut, f)
        
        json_size = os.path.getsize(pred_json_cut) / (1000.0 * 1000.0)
        print('json_size ', json_size)
        if json_size < 20:
            break
        
        conf_thres_try = round(conf_thres_try + 0.001, 3)

    print('')


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')

    parser.add_argument('--data', type=str, default='data/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1536, help='inference size (pixels)')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='final', help='save to project/name')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
