# dropCAM-pytorch

pytorch implementation of dropCAM (work in progress)

## Requirements 
- pip install pretrainedmodels
- pip install googledrivedownloader

## Usage
To run the gradCAM (vanilla) method
```
python test.py --gpu_number=0 --method=vanilla --model=vgg --runs_dir=vgg16bn &
```

To run the dropCAM (ours) method
```
python test.py --gpu_number=0 --method=ours --model=vgg --runs_dir=vgg16bn &
```

