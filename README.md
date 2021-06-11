<div align="center">    
 
# Deep Traffic Generation
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--  
Conference   
-->   
</div>
 
## Description   

Package that provides neural networks (mostly autoencoders) to embed and generate traffic trajectories.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/alafage/traffic-generation

# install project   
cd traffic-generation 
pip install -e .
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd traffic_generation

# example: run module
python linear_ae.py
# example: run module with arguments
python linear_ae.py --gpus 1 --early_stop --max_epochs 200
```

You can use Tensorboard to analyse your network's trainings

```bash
tensorboard --logdir lightning_logs
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from deep_traffic_generation.linear_ae import Linear_AE
from pytorch_lightning import Trainer

# model
model = LinearAE()

# data
train, val, test = ...

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
<!--
### Citation   
```
@article{Adrien Lafage,
  title={Your Title},
  author={Your team},
  journal={Location},
}
-->