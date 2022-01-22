## JSM Source Code

Source repository for our paper entilted "[Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection](https://proceedings.neurips.cc/paper/2021/file/642e92efb79421734881b53e1e1b18b6-Paper.pdf)" accepted by NeurIPS 2021 (poster).

------



## Pre-Implementation

1. **DenseCRF Installation**: Refer to DenseCRF Readme.md, and run demo successfully.
2. **Pytorch Environment**: Run ```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch```.
3. Run ```pip install tqdm```.
4. Run ```pip install pandas```.
5. Run ```pip install tensorboardX```.
6. Run ```pip install fairseq```. Possible Question "SyntaxError: invalid syntax", please see FAQ-Q1 below.
7. Run ```pip install scipy```.
8. Run ```pip install matplotlib```.

------


### Our JSM Implementation (Weakly Supervised)

1. Inference stage: ```python demo_test.py```; Using [Pre-trained Model](), e.g., ```ckpt_name = '.48'```.  
2. Train stage: ```CUDA_VISIBLE_DEVICES=2 python demo_train.py```                             
3. Check the log file: ```cat ./result.txt```
4. Load the training details: ```tensorboard --logdir=/YourComputer/JSM_model/runs/*```


### Dataset & Evaluation
1. The proposed **CapS** benchmark: you can [download]() directly, including initial pseudo-labels, captions, tags, and etc. More details are approached [in this link](). 
2. Widely-used RGB-D SOD benchmarks: you can [download]() directly. 
3. We use [this toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluating all SOD models.



### Bibtex
```
@InProceedings{li2021joint,
    author    = {Li, Jingjing and Ji, Wei and Bi, Qi and Yan, Cheng and Zhang, Miao and Piao, Yongri and Lu, Huchuan and Cheng, Li},
    title     = {Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection},
    booktitle = {NeurIPS},
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

