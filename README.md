# RMEUNet
There are some codes about RMEUNet that can be used for training and test.


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
## Datasets
### Synapse
You can refer to [this link](https://drive.google.com/file/d/1tGqMx-E4QZpSg2HQbVq5W3KSTHSG0hjK/view) to download the Synapse dataset and place it in **./data/synapse**.
### ACDC
You can refer to [this link](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) to download the Synapse dataset and place it in **./data/ACDC**.
### Polyp
You can refer to [this link](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view) to download the Synapse dataset and place it in **./data/polyp**.
## Training
```
python train_synapse.py
```
## Test
```
python test_synapse.py
```
