# RMEUNet
There is some codes about RMEUNet can be used for training and test.


The file organization structure should be：


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
