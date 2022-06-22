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
# # if using mAy-I trn-a -
# $ docker run --gpus '"device=0,1,3"' --name 2022AIChallenge --shm-size 8G -v ~/workspace/code:/root/workspace/code -v /hdd/a/data/DATA:/DATA -it nvcr.io/nvidia/pytorch:20.12-py3
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
~/workspace/code/2022AIChallenge
|-- data/
    |-- hyps/
	    |-- ...
	|-- dataset.yaml
|-- models
|-- utils
|-- predict.py
|-- README.md
|-- requirements.txt
|-- train.py
|-- val.py
```

### output 상세 설명

TBD

- - -

## 학습 및 추론 시간 제한

본 대회에서 제시된 재현 서버 사양은 다음과 같습니다.  

> 재현 서버 사양: 10C, Nvidia T4 GPU x 1, 90MEM, 1TB

그러나 본 대회를 준비함에 있어 사용한 내부 서버의 사양은 `GeForce RTX 2080 Ti`로 재현 서버 사양과 차이가 있으며, 사전에 위 재현 서버에서 테스트 해 볼 수가 없어, [구글 Colab](https://colab.research.google.com/?hl=ko)에서 기본적으로 제공하는 T4 환경에서 학습 및 추론 속도를 간단하게 비교해보았습니다.

- T4: 1epoch 기준 train 65초, val 46초
- 2080 Ti: 1epoch 기준 train 28초, val 17초

즉, T4에서 학습 시간 36시간, 추론 시간 3시간의 제한은 2080 Ti에서는 학습 시간 약 15.51시간, 추론 시간 1.11시간의 제한과 동일함을 의미합니다.

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