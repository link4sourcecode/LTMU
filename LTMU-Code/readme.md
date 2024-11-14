## Getting Started
### Requirements
All codes are written by Python 3.7 with
- PyTorch = 1.13.1
- torchvision = 0.14.1
- numpy = 1.21.5

### Preparing Datasets
Download the datasets CIFAR-10, CIFAR-100 to LTMU/data. The directory should look like

````
LTMU/data
├── CIFAR-100-python
└── CIFAR-10-batches-py
````
## Training
When training the original model with varying imbalance ratios, one can adjust the `imbalance_rate` parameter. Specifically, it can be set to values within the range of [0.01, 0.02, 0.1].

for CIFAR-10-LT
````
python train.py --dataset cifar10 -a resnet18 --num_classes 10 --imbanlance_rate 0.01 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3
````

for CIFAR-100-LT
````
python train.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.01 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 
````

To perform retraining from scratch and obtain the retrain model of the specified class that you want to forget, you can add the parameter `--forget_class forget_cls`.

## Unlearning
To execute our LTMU method, you can run the following code
````
python Unlearning.py --dataset cifar10 -a resnet18 --num_classes 10  --imbanlance_rate 0.01 --ori_path model_path --forget_class forget_cls --k number_of_k
````

To execute other baseline methods, you can run the corresponding code as follows. The method can be selected from the provided list which includes [BS, GA, RL, BT, FT, L1, GA_IA, GA_RFA, BS_IA, BS_RFA]

````
python Unlearning.py --dataset cifar10 -a resnet18 --num_classes 10  --imbanlance_rate 0.01 --ori_path model_path --forget_class forget_cls --unlearn_method method_name
````