## JSM Source Code

Source repository for our paper entilted "[Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection](https://proceedings.neurips.cc/paper/2021/file/642e92efb79421734881b53e1e1b18b6-Paper.pdf)" accepted by NeurIPS 2021 (poster).

------



## Pre-Implementation

1. ```vim ./DenseCRF/README.md ```
2. **DenseCRF Installation**: Refer to [DenseCRF Readme.md](https://github.com/jiwei0921/JSM/blob/main/DenseCRF/README.md), and run demo successfully.
3. **Pytorch Environment**: Run ```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch```.
4. Run ```pip install tqdm```.
5. Run ```pip install pandas```.
6. Run ```pip install tensorboardX```.
7. Run ```pip install fairseq```. Possible Question "SyntaxError: invalid syntax", please see FAQ-Q1 below.
8. Run ```pip install scipy```.
9. Run ```pip install matplotlib```.


### Dataset & Evaluation
1. The proposed **CapS dataset**: you can download directly ([Baidu Cloud (Passworde: 5okc)](https://pan.baidu.com/s/1IpqC2HTJzhfWDvqZpC2WTg) or [Google Drive](https://drive.google.com/file/d/1Oy9OGvQD2H7xrV9WH1j3n8xNS2UVT1cY/view?usp=sharing)), including initial pseudo-labels, captions, tags, and etc. More details are approached [in this link](https://proceedings.neurips.cc/paper/2021/file/642e92efb79421734881b53e1e1b18b6-Supplemental.pdf). 
2. RGB-D SOD benchmarks: you can [download](https://github.com/jiwei0921/RGBD-SOD-datasets) directly for realted RGBD SOD test sets. 
3. We use [this toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluating all SOD models.


------


### Our JSM Implementation (Weakly Supervised)

1. Modify the path of dataset in ```python demo_test.py``` and ```python demo_train.py```.
2. **Inference stage**: ```python demo_test.py```; Using Pre-trained Model ([Baidu Cloud (Passworde: vs85)](https://pan.baidu.com/s/1GbHR4V3jzqh1SGaQopIJGw) or [Google Drive](https://drive.google.com/file/d/1osp-8nEx_cAY9mjhaC9OJRTIQSP0irdr/view?usp=sharing)).  
3. **Training stage**: ```CUDA_VISIBLE_DEVICES=0 python demo_train.py```                             
4. Check the log file: ```cat ./result.txt```
5. Load the training details: ```tensorboard --logdir=/YourComputer/JSM_model/runs/*```


### Saliency Results

Our weakly-supervied saliency results can be approached in [Baidu Cloud (Passworde: m10a)](https://pan.baidu.com/s/1oPJjR2apBvnbUkmNokr3CQ) or [Google Drive](https://drive.google.com/file/d/1VwvTZFwRUtoEdymv5RzxywWuBmn4z7Xx/view?usp=sharing).
If you want to use our JSM to test on your own dataset, you can load our pretrained ckpt and run ```python demo_test.py``` directly.



### Bibtex
```
@InProceedings{li2021joint,
    author    = {Li, Jingjing and Ji, Wei and Bi, Qi and Yan, Cheng and Zhang, Miao and Piao, Yongri and Lu, Huchuan and Cheng, Li},
    title     = {Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection},
    booktitle = {Advances in Neural Information Processing Systems},
    month     = {December},
    year      = {2021}
}
```

### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).


---
+ #### FAQ

**Question1**: When installing ```fairseq```， post an 'SyntaxError: invalid syntax' 

Answer1: You can directly update python version, e.g., ```conda install python=3.7```. More details can be found [in this channel](https://github.com/pytorch/fairseq/issues/55).

**Question2**: You should replace the inplace operation by an out-of-place one. 

Answer2: This is because `*=` is not compatible with Python 3.9. `q *= self.scaling` -> `q = q * self.scaling`

