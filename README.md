# 🦷 ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings

<div align="center">
  <a href="https://arxiv.org/abs/2506.02702"><img src="https://img.shields.io/badge/ArXiv-2506.02702-red"></a> &ensp;
</div>

<div align="center">
  
[Tibor Kubik](https://scholar.google.com/citations?user=Zb6MSKcAAAAJ), [Francois Guibault](https://scholar.google.com/citations?user=KF8zbPUAAAAJ&hl=sk&oi=ao), [Michal Spanel](https://scholar.google.com/citations?hl=sk&user=75XIbgQAAAAJ) and [Herve Lombaert](https://scholar.google.com/citations?hl=sk&user=KQbyRzIAAAAJ)

![Diagram](assets/method-outline.png)

</div>


## 🔥 News and Todo
* 💻 June 2025: Preparation of this GitHub repository.
* 🗣️ May 2025: ToothForge hits the spotlight! Presented as an oral talk at [IPMI 2025](https://ipmi2025.org/), one of the leading venues for medical imaging research.
* 🎉 February 2025: ToothForge is officially accepted to [IPMI 2025](https://ipmi2025.org/) with a competitive 26% acceptance rate.

## Abstract
We introduce ToothForge, a spectral approach for automatically generating novel 3D teeth, effectively addressing the sparsity of dental shape datasets. By operating in the spectral domain, our method enables compact machine learning modeling, allowing the generation of high-resolution tooth meshes in milliseconds. However, generating shape spectra comes with the instability of the decomposed harmonics. To address this, we propose modeling the latent manifold on synchronized frequential embeddings. Spectra of all data samples are aligned to a common basis prior to the training procedure, effectively eliminating biases introduced by the decomposition instability. Furthermore, synchronized modeling removes the limiting factor imposed by previous methods, which require all shapes to share a common fixed connectivity. Using a private dataset of real dental crowns, we observe a greater reconstruction quality of the synthetized shapes, exceeding those of models trained on unaligned embeddings. We also explore additional applications of spectral analysis in digital dentistry, such as shape compression and interpolation. ToothForge facilitates a range of approaches at the intersection of spectral analysis and machine learning, with fewer restrictions on mesh structure. This makes it applicable for shape analysis not only in dentistry, but also in broader medical applications, where guaranteeing consistent connectivity across shapes from various clinics is unrealistic.

## Requirements
The code was tested on

* Ubuntu 24.04
* Python 3.12
* PyTorch 2.7.0
* 1 NVIDIA GPU with CUDA version 11.8 (the method is not memory heavy, at least when using 256 embeddings, so any gpu with at least 8GB will work).

### Setup an environment
```shell
conda create -n toothForge python==3.12
conda activate toothForge
```
### Install PyTorch (optional: if you plan to train a model, not just apply decomposition/alignment)
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Other Dependencies
```shell
pip install -r requirements.txt
```
Note: to avoid dependency conflicts, please make sure to use the exact package versions specified in `requirements.txt`. 

## Applying Spectral Decomposition
todo: add information about the first step: how to get the embeddings of the shapes.

## Data Preparation and Training
todo: describe how to generate synchronized embeddings and run training loop

## 🔗 BibTeX
If you find this work useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{kubik25toothforge,
      title={ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings}, 
      author={Tibor Kubik and Francois Guibault and Michal Spanel and Herve Lombaert},
      year={2025},
      eprint={2506.02702},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.02702}, 
}
```
