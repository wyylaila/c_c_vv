# RMEUNet
There is some codes about RMEUNet can be used for training and test.


The file organization structure should be：

```
RMEUNet/
├── data/
│   ├── ACDC/
│   ├── polyp/
│   └── synapse/
├── lists/ 
│   └──lists_Synapse/
├── model_pth/
├── result_map/
├── REUNet/
│   └── lib/
│       ├── ECANet.py
│       ├── RMT.py
│       ├── RepConv.py
│       ├── SCconv.py
│       ├── decoders.py
│       ├── networks.py
│       ├── pvtv2.py
│       └── resnet.py
├── test_log/
├── utils/
│   ├── dataloader.py
│   ├── dataset_synapse.py
│   ├── format_conversion.py
│   ├── preprocess_synapse_data.py
│   └── utils.py
├── README.md
├── test_synapse.py
├── train_synapse.py
└── trainer.py
```
## Some models
You should download the pretrained model of resnet from [this link](https://download.pytorch.org/models/).
And place it in  **./REUNet/lib/**.
## Synapse
### Dataset
You can refer to [this link](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) to download the Synapse dataset and place it in **./data/**.
### Train 
```
python train_synapse.py
```
### Test
```
python test_synapse.py
```
## ACDC and Polyop
The codes will be shown as soon.
