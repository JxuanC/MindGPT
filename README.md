# MindGPT: Interpreting What You See with Non-invasive Brain Recordings
Official Implementation of MindGPT in PyTorch

## News

* 2024-07-11
    
    Codes release.


* 2023-09-28
    
    Preprint release. Codes will be released soon!



## Overview
![MindGPT](assets/MindGPT.png)

## Samples
![brain2text results](assets/brain2text.png)

## Environment setup
1. `pip install -r requirements.txt`

2. Download [DIR dataset](https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577) (Kamitani Lab) and [ImageNet dataset](https://image-net.org/).

3. Extract CLIP visual representations by running `feature_extract.py` and use [SMALLCAP](https://github.com/RitaRamo/smallcap) to generate pseudo labels (format see `caption/example.json`).

4. Change Paths in `data/configure.py` to match your file locations.


## Training
Hyper-parameters can be changed with command line arguments
```
python brain2text_train.py --n_epochs 20 --batch_size 128
```

## Reconstruction with Trained Checkpoints
```
python brain2text_infer.py
```

## Acknowledgement
We thank Kamitani Lab for making their raw and pre-processed data public. Our MindGPT implementation is based on the [SMALLCAP](https://github.com/RitaRamo/smallcap). We thank these authors for making their codes and checkpoints publicly available!

## Cite
```
@article{chen2023mindgpt,
      title={MindGPT: Interpreting What You See with Non-invasive Brain Recordings}, 
      author={Jiaxuan Chen and Yu Qi and Yueming Wang and Gang Pan},
      year={2023},
      journal={arXiv preprint arXiv:2309.15729},
}
```
