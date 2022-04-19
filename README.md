>This repo is a **Ascend NPU** implement forkd from [SPACH](https://github.com/microsoft/SPACH).  
>To learn more about Ascend NPU, [click here](https://gitee.com/ascend/modelzoo).

This repository contains Pytorch evaluation code, training code and pretrained models for the following projects:

+ SPACH ([A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP](https://arxiv.org/abs/2108.13002))
+ sMLP ([Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?](https://arxiv.org/abs/2109.05422))
+ ShiftViT ([When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism](https://arxiv.org/abs/2201.10801))

Other unofficial implementations:

+ ShiftViT
  + [Keras](https://keras.io/examples/vision/shiftvit/) by [Aritra Roy Gosthipaty](https://twitter.com/ariG23498) and [Ritwik Raha](https://twitter.com/ritwik_raha)

# Main Results on ImageNet with Pretrained Models


| name               | acc@1 | #params | FLOPs | url                                                          |
| ------------------ | ----- | ------- | ----- | ------------------------------------------------------------ |
| SPACH-Conv-MS-S    | 81.6  | 44M     | 7.2G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_conv_s.pth) |
| SPACH-Trans-MS-S   | 82.9  | 40M     | 7.6G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_trans_s.pth) |
| SPACH-MLP-MS-S     | 82.1  | 46M     | 8.2G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_mlp_s.pth) |
| SPACH-Hybrid-MS-S  | 83.7  | 63M     | 11.2G | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_hybrid_s.pth) |
| SPACH-Hybrid-MS-S+ | 83.9  | 63M     | 12.3G | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_hybrid_s+.pth) |
| sMLPNet-T          | 81.9  | 24M     | 5.0G  |                                                              |
| sMLPNet-S          | 83.1  | 49M     | 10.3G | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/smlp_s.pth) |
| sMLPNet-B          | 83.4  | 66M     | 14.0G | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/smlp_b.pth) |
| Shift-T / light    | 79.4  | 20M     | 3.0G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/shiftvit_tiny_light.pth) |
| Shift-T            | 81.7  | 29M     | 4.5G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/shiftvit_tiny_r2.pth) |
| Shift-S / light    | 81.6  | 34M     | 5.7G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/shiftvit_small_light.pth) |
| Shift-S            | 82.8  | 50M     | 8.8G  | [github](https://github.com/microsoft/SPACH/releases/download/v1.0/shiftvit_small_r2.pth) |

# Usage

## NPU
NPU means neural-network processing units. 

To enable NPU, use parser argument
>**--npu**

## Install
First, clone the repo and install requirements:

```bash
git clone https://github.com/microsoft/Spach
```

+ install requirement
```
pip3 install torchvision==0.6.0
pip3 install einops==0.4.1
pip3 install --no-deps timm==0.4.5

# other recommended requirements
apex==0.1+ascend.20220315
torch==1.5.0+ascend.post5.20220315
```
- source env and build：

```
source ./test/env_npu.sh
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. 
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), 
and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training & Evaluation

One can simply call the following script to run training process. Distributed training is recommended even on single GPU node. 

To train a model on NPU, run `main.py` with the desired model architecture and the path to the ImageNet dataset:


```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --model=model_name \
--data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --model=model_name \
--data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --model=model_name \
--data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --model=model_name \
--data_path=real_data_path

#evaluate 8p accuracy
bash ./test/train_eval_8p.sh --model=model_name \
--data_path=real_data_path \
--resume=checkpoint_path
```

# Citation

```
@article{zhao2021battle,
  title={A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP},
  author={Zhao, Yucheng and Wang, Guangting and Tang, Chuanxin and Luo, Chong and Zeng, Wenjun and Zha, Zheng-Jun},
  journal={arXiv preprint arXiv:2108.13002},
  year={2021}
}

@article{tang2021sparse,
  title={Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?},
  author={Tang, Chuanxin and Zhao, Yucheng and Wang, Guangting and Luo, Chong and Xie, Wenxuan and Zeng, Wenjun},
  journal={arXiv preprint arXiv:2109.05422},
  year={2021}
}

```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Acknowledgement

Our code are built on top of [DeiT](https://github.com/facebookresearch/deit). We test throughput following [Swin Transformer](https://github.com/microsoft/Swin-Transformer)