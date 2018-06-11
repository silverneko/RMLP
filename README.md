# RMLP
> **Multi-focus image fusion via region mosaicking on Laplacian pyramid.**  
> _Jianguo Sun, Qilong Han, Liang Kou, Liguo Zhang, Kejia Zhang, and Zilong Jin_


## Checklist
- [ ] Photo data
- [ ] Synthetic data
- [ ] Public dataset to compare with other approaches
- [ ] Algorithm implementation
- [ ] Video demo


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for evaluation.

### Prerequisites
You should have a [functional `conda` environment](https://docs.anaconda.com/anaconda/install/).

### Building and Running
Assuming you are in the repository root directory, create an environment by
```
conda env create -f environment.yml
```
This will create an environment called `rmlp`, activate it with
```
conda activate rmlp
```
Use
```
python demo.py
```
to run the demo script.


## Authors
TBA


## License
TBA


## References
Remember to cite the following works if any of the datasets is used.
- [Slavica Savic, "Multifocus Image Fusion Based on Empirical Mode Decomposition", Twentieth International Electrotechnical and Computer Science Conference, ERK 2011](http://dsp.etfbl.net/mif/)
    - Most of the image pairs are not registered.
    - No ground-truth.
    - Usable but not ideal.
- [Lytro Multi-focus Image Dataset](https://www.researchgate.net/publication/291522937_Lytro_Multi-focus_Image_Dataset)
    - All image pairs are registered.
    - No ground-truth.
- [A classification and fuzzy-based approach for digital multi-focus image fusion August 2011 Jamal Saeedi · Karim Faez](https://www.researchgate.net/publication/273000238_multi-focus_image_dataset)
    - Some of the image pairs are not registered.
    - Some of the image pairs have ground-truth.
