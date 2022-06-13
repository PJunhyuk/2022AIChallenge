# 2022AIChallenge

## 셋업

학습 및 추론을 위한 환경을 구축하는 단계입니다.  

### 별도 셋업

별도의 환경을 위한 셋업 과정입니다. docker 가 설치되어 있고, dataset 이 알맞은 경로에 준비되어 있다면 생략할 수 있습니다.  

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

#### 과제 데이터 다운로드 및 셋업

제공된 데이터는 다음과 같은 경로로 셋업되어야 합니다.  

```
/DATA
  /train
    /images
    /labels
  /test
    /images
    Test_Images_Information.json
```

위와 같이 셋업되어 있지 않은 경우, 제시된 데이터 파일을 다운로드 받아 다음의 코드로 세팅합니다.

```bash
$ sudo mkdir -p /DATA/train
$ sudo unzip train.zip -d /DATA/train
$ sudo rm train.zip

$ sudo mkdir -p /DATA/test
$ sudo unzip test.zip -d /DATA/test
$ sudo rm test.zip

$ sudo mv Test_Images_Information.json /DATA/test
```

### 작업 폴더 세팅

작업 폴더를 세팅하기 위해 제출한 코드를 `~/workspace/code/2022AIChallenge` 에 세팅합니다.  
혹은 다음과 같이 `git` 에서 가져옵니다.  

```bash
$ mkdir -p ~/workspace/code
(~/workspace/code) $ git clone https://github.com/PJunhyuk/2022AIChallenge
```

** 이후의 모든 코드는 특별한 언급이 없다면 current work directory(`~/workspace/code/2022AIChallenge`) 하에서의 실행을 전제합니다.  

### docker 및 git, ffmpeg (for opencv) 세팅

여러 docker image 중 `nvidia/pytorch` 의 기본 이미지를 활용하였습니다. 다음과 같은 방식으로 docker 를 가져오고, 기본 package 인 git 과 ffmpeg 를 설치합니다.  
* 추가 설치가 워낙 간단하여, 별도로 docker image 파일을 만들지는 않았습니다.  

```bash
$ docker pull nvcr.io/nvidia/pytorch:20.12-py3
$ docker run --gpus all --name 2022AIChallenge --shm-size 8G -v ~/workspace/code:/root/workspace/code -v /DATA:/DATA -it nvcr.io/nvidia/pytorch:20.12-py3
# # if using mAy-I trn-a -
# $ docker run --gpus '"device=0,1,3"' --name 2022AIChallenge --shm-size 8G -v ~/workspace/code:/root/workspace/code -v /hdd/a/data/DATA:/DATA -it nvcr.io/nvidia/pytorch:20.12-py3

# Install git & ffmpeg
# 'glib2' is a dependency of 'opencv'
# type 6-69-6
$ apt-get update && apt-get install -y --no-install-recommends \
    git libxrender1 ffmpeg libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
```

### dependencies 설치

```bash
$ pip install -r requirements.txt
```

- - -

## 학습 및 추론

```bash
$ python dataset_prepare.py
```

### Train

```bash
# homeUBT
$ python train.py --batch 4 --cfg yolov5s.yaml --weights yolov5s --data dataset.yaml --img 640 --epochs 20
# inf-a
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6 --data dataset.yaml --img 1280 --epochs 20
$ python train.py --batch 16 --cfg yolov5s.yaml --weights yolov5s --data dataset.yaml --img 640
# inf-b
$ python train.py --batch 16 --cfg yolov5s.yaml --weights yolov5s --data dataset.yaml --img 640


# trn-a
$ python train.py --batch 4 --hyp data/hyps/hyp.scratch-high.yaml --cfg yolov5l6.yaml --weights yolov5l6 --data dataset.yaml --img 1280
# inf-a
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6 --data dataset.yaml --img 1280
# inf-b
$ python train.py --batch 16 --cfg yolov5s.yaml --weights yolov5s --data dataset.yaml --img 640

# inf-a exp7 6/10 19:30
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6 --data dataset_diet.yaml --img 1280
# inf-b exp9 6/10 22:45
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6.pt --data dataset_diet.yaml --img 1280
# inf-a exp8
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6.pt --data dataset.yaml --img 1280
# trn-a exp 6/11 02:55
$ python train.py --batch 9 --cfg yolov5m6.yaml --weights yolov5m6.pt --hyp data/hyps/hyp.scratch-high.yaml --data dataset.yaml --img 1280
# inf-b exp10 6/11 10:17
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6.pt --data dataset.yaml --img 1280 --image-weights
# inf-a 
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6.pt --data dataset.yaml --hyp data/hyps/hyp.VOC.yaml --img 1280 --image-weights
$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6.pt --data dataset.yaml --img 1280 --image-weights --epoch-parts 5

$ python train.py --batch 4 --cfg yolov5s6.yaml --weights yolov5s6.pt --data dataset.yaml --img 1280 --image-weights --epoch-parts 5
```

### Val

```bash
$ python val.py --batch 16 --weights runs/inf-a/exp4/weights/last.pt --data data/dataset.yaml --img 640 --task val --conf-thres 0.1
$ python val.py --batch 16 --weights runs/inf-a/exp4/weights/last.pt --data data/dataset.yaml --img 640 --task val --conf-thres 0.005
```

### Predict

```bash
$ python test_txt.py
```

```bash
$ python predict.py --weights best.pt --data data/dataset.yaml --imgsz 1280 --source /DATA/test/images --nosave

$ python val.py --batch 4 --weights best.pt --data data/dataset.yaml --img 1280 --task test --save-json --conf-thres 0.1

$ python val.py --batch 16 --weights runs/inf-a/exp4/weights/last.pt --data data/dataset.yaml --img 640 --task test --save-json --conf-thres 0.1
$ python val.py --batch 16 --weights runs/inf-b/exp5/weights/last.pt --data data/dataset.yaml --img 640 --task test --save-json --conf-thres 0.1

# inf-a
$ python val.py --batch 4 --weights runs/inf-a/exp6/weights/last.pt --data data/dataset.yaml --img 1280 --task test --save-json --conf-thres 0.1

# inf-b
$ python val.py --batch 4 --weights runs/inf-b/exp6/weights/last.pt --data data/dataset.yaml --img 1280 --task test --save-json --conf-thres 0.1

```