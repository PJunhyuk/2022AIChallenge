# 2022AIChallenge

본 repo는 *마이애미* 팀의 2022 인공지능 온라인 경진대회 중 *주차 관련 이동체 객체 검출 문제* 태스크 수행을 위한 레포지토리입니다.  

- - -

## 셋업

학습 및 추론을 위한 환경을 구축하는 단계입니다.  

### 별도 셋업

별도의 환경을 위한 셋업 과정입니다. 재현성 검증 서버에서는 가상환경 설정 없이 직접 라이브러리를 설치하여 사용하기 때문에 본 과정은 생략합니다.  
다른 서버라도 docker 가 설치되어 있고, dataset 이 알맞은 경로에 준비되어 있다면 생략할 수 있습니다.  

#### docker 설치

본 repo 는 간편한 설치를 위해 `docker` 를 사용합니다. 서버에 `docker` 가 설치되어 있지 않은 경우 다음과 같은 방식으로 설치 가능합니다.  

```bash
$ sudo apt-get remove docker docker-engine docker.io
$ sudo apt-get update && sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

$ sudo apt-get update && sudo apt-cache search docker-ce
# Message: docker-ce - Docker: the open-source application container engine

$ sudo apt-get update && sudo apt-get install docker-ce
$ sudo usermod -aG docker $USER

$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

도중에 `sudo: unable to resolve host` 에러가 나오면 [링크](https://extrememanual.net/33739) 로 해결하면 됩니다.

#### docker 및 git, ffmpeg (for opencv) 세팅

여러 docker image 중 `nvidia/pytorch` 의 기본 이미지를 활용합니다. 다음과 같은 방식으로 docker 를 가져오고, 기본 package 인 git 과 ffmpeg 를 설치합니다.  
* 추가 설치가 워낙 간단하여, 별도로 docker image 파일을 만들지는 않았습니다.  

```bash
$ docker pull nvcr.io/nvidia/pytorch:20.12-py3
$ docker run --gpus all --name 2022AIChallenge --shm-size 8G -v ~/workspace/code:/root/workspace/code -v /DATA:/DATA -it nvcr.io/nvidia/pytorch:20.12-py3
```

#### 과제 데이터 다운로드 및 셋업

제공된 데이터는 `/DATA` 디렉토리에 다음과 같은 형태로 준비되어있음을 전제합니다. (재현성 검증 서버 기준)  

```
/DATA
|-- test/
    |-- images/
        |-- 0dbc1884-9895-4294-91ef-77626a5ca826.png
        |-- ...
|-- train/
    |-- images/
        |-- 20201102_경기도_-_-_맑음_주간_실외_right_000079_0088055.png
        |-- ...
    |-- label/
        |-- Train.json
|-- sample_submission.json
|-- Test_Images_Information.json
|-- test.zip
|-- train.zip
```

### 세부 환경 세팅

```bash
# Install git & ffmpeg
# 'glib2' is a dependency of 'opencv'
# type 6-69-6
$ apt-get update && apt-get install -y --no-install-recommends \
    git libxrender1 ffmpeg libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
```

도중에 `GPG error` 가 발생하면 [링크](https://eehoeskrap.tistory.com/454) 로 해결하면 됩니다.  

#### 작업 폴더 세팅

작업 폴더를 세팅하기 위해 제출한 코드를 `/USER` 디렉토리에 세팅합니다.  
혹은 다음과 같이 `git` 에서 가져옵니다.  

```bash
(/USER) $ git clone https://github.com/PJunhyuk/2022AIChallenge
```

** 이후의 모든 코드는 특별한 언급이 없다면 current work directory(`/USER/2022AIChallenge`) 하에서의 실행을 전제합니다.  


### dependencies 설치

재현성 검증 서버에서는 이후 모든 코드를 Jupyter 노트북 Terminal 에서의 실행을 전제합니다.  

```bash
$ pip install -r requirements.txt
```

- - -

## 학습 및 추론

### 학습

```bash
$ python train.py
```

### 추론

```bash
$ python predict.py
```

- - -

## 상세 설명

repo 전반에 대한 상세 설명입니다.  

### 코드 상세 설명

```
/USER/2022AIChallenge
|-- data/
    |-- hyps/
	    |-- hyp.finetune.yaml
        |-- hyp.scratch-low.yaml
	|-- dataset.yaml
|-- models/
    |-- ...
|-- utils/
    |-- ...
|-- predict.py
|-- README.md
|-- requirements.txt
|-- train.py
|-- val.py
```

### `submission` 폴더 설명

- `baseline_last.pt` : yolov5x6.pt 에서 hyp.scratch-low.yaml 을 기반으로 50 epoch 학습한 모델 가중치 파일입니다.
- `tune_last.pt` : baseline_last.pt 에서 hyp.finetune.yaml 을 기반으로 15 epoch 추가 학습한 모델 가중치 파일입니다. *최고점 제출물에 대응하는 모델 가중치 파일입니다.*
- `best_preds_cut.json` : *최고점 Submission 파일입니다.*

다음의 명령어를 통해 `tune_last.pt` 에서 `best_preds_cut.json` 을 직접 생성할 수 있습니다.  

```bash
$ python predict.py --weights submission/tune_last.pt
```

### `train.py` 설명

모델 학습에 사용되는 python 파일입니다.  

#### 1. dataset 폴더 생성 및 세팅

`train.py` 를 실행하면 우선 학습에 사용할 데이터셋 폴더를 생성하고 세팅하는 절차가 진행됩니다. `train.py` 의 `data_prepare()` 함수를 사용합니다.  
`/DATA` 폴더의 데이터를 읽어 `../dataset/` 폴더에 학습에 적합한 형태로 이미지를 복사하여 세팅하고, 학습에 적합한 형태로 label 파일들을 생성합니다.  
재현성 검증 서버 기준 30분 정도 소요됩니다.  

```
data preparing
generate raw_train.json, raw_val.json
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134741/134741 [00:24<00:00, 5574.02it/s]
generate dataset/train, dataset/val
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16258/16258 [04:45<00:00, 56.95it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118483/118483 [28:13<00:00, 69.97it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19521/19521 [00:00<00:00, 1632512.03it/s]
```

#### 2. baseline 학습

우선 yolov5x6.pt pre-trained weight 을 사용하여 baseline 학습을 진행합니다. flag 없이 실행한다면 다음의 주요 args 들이 default 로 설정되어 있습니다.  

```
--weights yolov5x6.pt
--epochs 50
--epoch-parts 15
--batch-size 2
--image-weights True
--imgsz 1280
--hyp data/hyps/hyp.scratch-low.yaml
```

재현성 검증 서버 기준 한 epoch 학습에 30분 정도 소요됩니다. 50 epoch 을 학습하기 때문에 전체로는 26시간 정도 소요됩니다.  

```
train: weights=yolov5x6.pt, data=data/dataset.yaml, epochs=50, epoch_parts=15, batch_size=2, no_image_weights=False, imgsz=1280, hyp=data/hyps/hyp.scratch-low.yaml, val_period=0, no_data_prepare=False, path_DATA_dir=/DATA, project=runs/train, name=final, cfg=, rect=False, resume=False, nosave=False, noautoanchor=False, noplots=False, bucket=, cache=None, device=, multi_scale=False, optimizer=SGD, sync_bn=False, workers=8, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, local_rank=-1, image_weights=True, noval=True
hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt to yolov5x6.pt...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 270M/270M [00:52<00:00, 5.41MB/s]
Scaled weight_decay = 0.0005
optimizer: SGD with parameter groups 159 weight (no decay), 163 weight, 163 bias

AutoAnchor: 6.10 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 1280 train, 1280 val
Using 2 dataloader workers
Logging results to runs/train/final
Starting training for 50 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      0/49     14.1G   0.07297   0.07933   0.05713        29      1280: 100%|██████████| 721/721 [30:20<00:00,  2.52s/it]                                                                                                                               

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      1/49     14.1G    0.0594   0.04674   0.04433        18      1280: 100%|██████████| 721/721 [30:23<00:00,  2.53s/it]                                                                                                                               
```

flag 없이 실행한다면 output 은 다음과 같은 형태로 생성됩니다.

```
/USER/2022AIChallenge
|-- runs/
    |-- train/
        |-- official/
            |-- weights/
                |-- best.pt
                |-- last.pt
            |-- ...
```

#### 3. finetuning 학습

우선 위 baseline 학습을 통해 생성된 last.pt 를 pre-trained weight 으로 사용하여 finetuning 학습을 진행합니다. flag 없이 실행한다면 다음의 주요 args 들이 default 로 설정되어 있습니다.  

```
--weights runs/train/official/weights/last.pt
--epochs-tune 15
--epoch-parts 15
--batch-size 2
--image-weights False
--imgsz 1280
--hyp-tune data/hyps/hyp.finetune.yaml
```

재현성 검증 서버 기준 한 epoch 학습에 30분 정도 소요됩니다. 15 epoch 을 학습하기 때문에 전체로는 8시간 정도 소요됩니다.  

flag 없이 실행한다면 output 은 다음과 같은 형태로 생성됩니다.

```
/USER/2022AIChallenge
|-- runs/
    |-- train/
        |-- official2/
            |-- weights/
                |-- best.pt
                |-- last.pt
            |-- ...
```

최종 생성된 `runs/train/official2/weights/best.pt` 파일이 최고점 제출물에 대응하는 모델 가중치 파일입니다.  

위 과정들을 모두 포함한 학습에 소요된 총 시간은 재현성 검증 시간 기준 34.5시간 정도로, 36시간 제한을 충족합니다.  

### `predict.py` 설명

모델 추론에 사용되는 python 파일입니다.  

#### 1. 모델 추론

`predict.py` 를 실행하면 우선 주어진 weights 로 추론을 진행합니다. flag 없이 실행한다면 다음의 주요 args 들이 default 로 설정되어 있습니다.  

```
--weights runs/train/official2/weights/best.pt
--batch-size 16
--iou-thres 0.7
--imgsz 1536
```

#### 2. conf cut

TBD

- - -

## Reproducibility

본 repo에서는 `utils/general.py`에서 `init_seeds` 함수를 통해 Reproducibility를 제어합니다.

```python
def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
```

그러나 PyTorch는 공식적으로 완벽히 Reproducibility를 제어할 수 없다고 합니다. 대표적으로 CUDA 함수를 사용하는 PyTorch 함수들 중 nondeterministic한 함수들이 존재합니다. 본 repo는 이 중 불가피하게 `torch.nn.funcional.interpolate()`를 사용하고 있어, 완벽한 Reproducibility 제어가 불가합니다.
- 레퍼런스: [Reproducible PyTorch를 위한 randomness 올바르게 제어하기!](https://hoya012.github.io/blog/reproducible_pytorch/)

실제로 같은 조건에서 학습을 진행해도 조금씩 다르게 계산되는 모습을 확인할 수 있었습니다.
- 위에 언급한 `torch.nn.funcional.interpolate()` 함수, 혹은 obj loss를 계산하는 과정에서 연산되는 `bcewithlogitsloss`에서 Reproducibility가 깨지는 것으로 추정됩니다.

때문에 본 repo에서는 완벽한 Reproducibility가 구현되어 있지는 않은 점을 감안 부탁드립니다. 그러나 위의 작업들로 최대한의 Reproducibility는 확보하여, 불가피한 정말 작은 차이들만이 존재합니다.  

- - -

## Reference

- [yolov5](https://github.com/ultralytics/yolov5)