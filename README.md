# DSE
This repository contains the code for the paper "**Discriminative Suprasphere Embedding for Fine-Grained Visual Categorization**"

## Requirements:  
  - Install Torch with CUDA GPU     
   Torch {http://torch.ch/docs/getting-started.html }  
  - Install cuDNN and the Torch cuDNN bindings  
   cuDNN bindings {https://github.com/soumith/cudnn.torch/tree/R4 }    
  - **Notes:** For convenience, we recommend using [docker](https://hub.docker.com/)
  
## CUB-200-2011 dataset and pretrained ResNet-101 model:  
  - Download CUB-200-2011 {http://www.vision.caltech.edu/visipedia/CUB-200-2011.html }  
  - Download pretrained ResNet-101 model {https://github.com/facebook/fb.resnet.torch/tree/master/pretrained }  
  - Arrange the dataset so that it contains a `\train` and a `\val` directory, which each contain sub-directories for every label. For example:  
      ```bash
      "train/<label1>/<image.jpg>  
       train/<label2>/<image.jpg>  
       val/<label1>/<image.jpg>  
       val/<label2>/<image.jpg>"  
       ```
   - To achieve this step you can use these files,  
      ```bash
      \tmp\dataset\move.py    
      \tmp\dataset\train_images.txt    
      \tmp\dataset\test_images.txt    
      ```
  - Place the rearranged dataset in `tmp\dataset\CUB`  
  - Place the pretrained ResNet-101 model in `tmp\models`    
  
## Training:  
  - Run the script inside `script\script.txt`    
  - Get the dataset index file `tmp\dataset\cubsphere.t7`  
  - Get the fine-tuned model `tmp\result\cub_fine_tuned_model\model_best.t7`  
  - Obtain classification accuracy    
  
## Visualization:  
  - Place the original downloaded dataset in `visualization\data\CUB_200_2011`  
  - Place the dataset index file `tmp\dataset\cubsphere.t7` in `visualization\data`  
  - Place the fine-tuned model `tmp\result\cub_fine_tuned_model\model_best.t7` in `visualization\model`    
  - Run the scripts under `visualization\` in sequence according to the file name number  
  - Obtain Phase Activation Map (PAM), Class Contribution Map (CCM), Mean IoU, Discriminative localization results.  
  
