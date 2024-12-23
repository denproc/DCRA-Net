

# DCRA-Net - Dynamic Cardiac Reconstruction Attention Network
[![arXiv](https://img.shields.io/badge/arXiv-2412.15342-b31b1b.svg)](https://arxiv.org/abs/2412.15342)

Pytorch implementation of DCRA-Net presented for dynamic fetal cardiac MRI reconstruction in

[DCRA-Net: Attention-Enabled Reconstruction Model for Dynamic Fetal Cardiac MRI](https://arxiv.org/abs/2412.15342)<br>
[Denis Prokopenko](https://www.linkedin.com/in/denproc/)<sup>1</sup>, [David F.A. Lloyd](https://www.kcl.ac.uk/people/david-f-a-lloyd)<sup>1,2</sup>, [Amedeo Chiribiri](https://www.kcl.ac.uk/people/amedeo-chiribiri)<sup>1</sup>, [Daniel Rueckert](https://aim-lab.io/author/daniel-ruckert/)<sup>3,4</sup>, [Joseph V. Hajnal](https://www.kcl.ac.uk/people/jo-hajnal)<sup>1</sup><br>
<sup>1</sup>King’s College London,<sup>2</sup>Evelina London Children’s Hospital, <sup>3</sup>Imperial College London, <sup>4</sup>Technical University of Munich



## Summary

Dynamic Cardiac Reconstruction Attention Network (DCRA-Net) - a model that reconstructs the dynamics of the fetal heart from highly accelerated free-running (non-gated) MRI acquisitions by taking advantage of attention mechanisms in spatial and temporal domains and temporal frequency representation of the data.


## Getting Started

### Installation
Clone repository and prepare the environment.
```bash
git clone https://github.com/denproc/DCRA-Net.git
cd DCRA-Net
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```

Download VISTA masks, sample data, and model checkpoints used in the paper.
```bash
# download VISTA masks (TBA)
curl -O https://link.to.masks
# unzip masks (TBA)
tar -xzvf masks.tar.gz
# download data samples (TBA)
curl -O https://link.to.sample_data
# unzip data samples (TBA)
tar -xzvf sample_data.tar.gz  
# download checkpoints (TBA)
curl -O https://link.to.checkpoints
# extract checkpoints (TBA)
tar -xzvf model_checkpoints.tar.gz  
```

### Usage


In this section, we demonstrate the use of DCRA-Net on a fetal cardiac MRI dataset with sequences truncated to 32 frames, resized, and center-cropped to a resolution of $96 \times 96$ pixels.
Application to adult cardiac MRI follows the same procedure, except that sequences are limited to 20 frames and resized to $160 \times 160$ pixels.
While a broader application of DCRA-Net is beyond the scope of this paper, the model could be adapted to other dynamic MRI domains, provided sufficient training data and computational resources are available.

To explore oprions the scripts:
```bash
python3 test.py -h
python3 tratin.py -h
```


Evaluation using pretrained model on data in `DATADIR`.
```bash
# Lattice Underasmpling
python3 test.py --backbone DCRA-Net --dc_mode force --image_size 96 --n_frames 32 --representation_time frequency --in_channels 2 --out_channels 2 --batch_size 1 --save_dir ./data/evaluation_fetal_lattice --acceleration 8 --pattern lattice --data_dir DATADIR --checkpoint_path ./data/model_checkpoints/dcranet_fetal_32-96-96_8x_lattice_checkpoint.pt --verbose

# VISTA Undersampling
python3 test.py --backbone DCRA-Net --dc_mode force --image_size 96 --n_frames 32 --representation_time frequency --in_channels 2 --out_channels 2 --batch_size 1 --save_dir ./data/evaluation_fetal_vista --acceleration 8 --pattern vista --mask_ucoef 07 --mask_dir ./data/vista_masks/96x32_acc8_07  --data_dir DATADIR --checkpoint_path ./data/model_checkpoints/dcranet_fetal_32-96-96_8x_vista_checkpoint.pt --verbose 
```

Training on you training data in `TRAINDATADIR`.
---
**NOTE**

The data is expected to be in a form of k-space sequences and to be stored as a directory of files.
The filename format is `{patient_id}_{other-details}.hdf5`. 
It is important to have the same `patient_id` for sequences acquired from the same subject for valid `train|val` split.

---


```bash
# Lattice Undersampling
python3 train.py --backbone DCRA-Net --dc_mode force --image_size 96 --n_frames 32 --representation_time frequency --in_channels 2 --out_channels 2 --batch_size 1 --start_epoch 0 --n_epochs 10 --save_dir ./data/new_version --acceleration 8 --pattern lattice --data_dir TRAINDATADIR --verbose

# VISTA Undersmapling
python3 train.py --backbone DCRA-Net --dc_mode force --image_size 96 --n_frames 32 --representation_time frequency --in_channels 2 --out_channels 2 --batch_size 1 --start_epoch 0 --n_epochs 10 --save_dir ./data/new_version --acceleration 8 --pattern vista --mask_ucoef 07 --mask_dir ./data/vista_masks/96x32_acc8_07  --data_dir TRAINDATADIR --verbose

```

## Citation

If you use DCRA-Net in your project or find it useful, please, cite our paper as follows.

```
@article{prokopenko2024dcranet,
  title={{DCRA-Net: Attention-Enabled Reconstruction Model for Dynamic Fetal Cardiac MRI}},
  author={Prokopenko, Denis and Lloyd, David FA and Chiribiri, Amedeo and Rueckert, Daniel and Hajnal, Joseph V},
  journal={arXiv preprint arXiv:2412.15342},
  year={2024},
}

```
