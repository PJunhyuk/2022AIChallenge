
### DATA 폴더 준비

```bash
$ sudo mkdir -p /DATA/train
$ sudo unzip train.zip -d /DATA/train
$ sudo rm train.zip
```

### docker 및 환경 세팅

```bash
$ docker run --gpus all --name 2022AICompetition --shm-size 8G -v ~/workspace/code:/root/workspace/code -v /DATA:/DATA -it nvcr.io/nvidia/pytorch:20.12-py3

# Install git & ffmpeg
# 'glib2' is a dependency of 'opencv'
# type 6-69-6
$ apt-get update && apt-get install -y --no-install-recommends \
    git libxrender1 ffmpeg libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

$ pip install -r requirements.txt
```

```bash
$ python dataset_prepare.py
```

### Train

```bash
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6 --data dataset.yaml --img 1280 --epochs 20
```