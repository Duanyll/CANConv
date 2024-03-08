# Dataset format

## Train and val datasets

Train and val datasets are h5 files, with the following structure

```
dataset.h5
    /gt:  N*C*64*64 FP64 Tensor [0, 2047)
    /lms: N*C*64*64 FP64 Tensor [0, 2047)
    /ms:  N*C*16*16 FP64 Tensor [0, 2047)
    /pan: N*1*64*64 FP64 Tensor [0, 2047)
```

where `N` is the amount of training images and `C` denotes the spectral num (8 or 4). 

## Module input and output

When training, the `SimplePanTrainer.forward` method accepts a `data` object with the following properties

```
data
    gt:  B * C * H    * W    FP32 Tensor [0, 1)
    lms: B * C * H    * W    FP32 Tensor [0, 1)
    ms:  B * C *(H/4) *(W/4) FP32 Tensor [0, 1)
    pan: B * 1 * H    * W    FP32 Tensor [0, 1)
```

The `SimplePanTrainer.forward` method picks required properties and pass them to the module. The predicted result of the module `sr` is a tensor

```
sr: B*C*H*W FP32 Tensor [0, 1)
```

## Testing datasets

Reduced test datasets file:

```
reduced_dataset.h5
    /gt:  N*C*256*256 FP64 Tensor [0, 2047)
    /lms: N*C*256*256 FP64 Tensor [0, 2047)
    /ms:  N*C* 64* 64 FP64 Tensor [0, 2047)
    /pan: N*1*256*256 FP64 Tensor [0, 2047)
```

OrigScale dataset file:

```
full_dataset.h5
    /lms: N*C*512*512 FP64 Tensor [0, 2047)
    /ms:  N*C*128*128 FP64 Tensor [0, 2047)
    /pan: N*1*512*512 FP64 Tensor [0, 2047)
```