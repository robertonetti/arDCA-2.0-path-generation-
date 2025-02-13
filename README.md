[![Paper](https://img.shields.io/badge/Published-Nat_Commun-green)](https://www.nature.com/articles/s41467-021-25756-4)
[![Licence](https://img.shields.io/badge/Licence-Apache-red)](https://www.nature.com/articles/s41467-021-25756-4)

# arDCA
auto-regressive Direct Coupling Analysis (arDCA) 2.0.

## Overview
This package is the GPU-accelerated version of the original version that can be found at [ArDCA.jl](https://github.com/pagnani/ArDCA.jl). The current implementation also aims at providing a user-friendly command line interface for training and sampling from an autoregressive DCA model.

## Installation
During the installation, `arDCA` will also install `adabmDCA` and all its dependencies.
### Option 1:
Install the package via PyPl:
```bash
python -m pip install arDCA
```
### Option 2
Clone this repository:
```bash
git clone https://github.com/spqb/arDCA.git
cd arDCA
python -m pip install .
```
## Using the package
We provide a Colab notebook where it is shown hot to train and sample an `arDCA` model using RNA sequences.

Alternatively, one can install the package locally and run from the command line one of the two implemented routines:

### Train arDCA from the command line
Once installed, you can launch the package routing by using the command `arDCA`. All the training options can be listed via
```bash
arDCA train -h
```
To launch a training with default arguments, use
```bash
arDCA train -d <path_data> -o <output_folder> -l <label>
```
where `path_data` is the path to the input multi-sequence alignment in [fasta](https://en.wikipedia.org/wiki/FASTA_format) format and `label` is an identifier for the output files. The parameters of the trained model are saved in the file `output_folder/<label>_params.pth`, and can be easily loaded afterwrds using the Pytorch methods.

By default, the program assumes that the input data are protein sequences. If you want to use RNA sequences, you should use the argument `--alphabet rna`.

> [!WARNING]
> Depending on the dataset, the default regularization parameters `reg_h` and `reg_J` may not work properly. If the training does not converge or the model's generation capabilities are poor, you may want to increase these values.

### Sample arDCA from the command line
To generate new sequences using the command line, the minimal input command is
```bash
arDCA sample -p <path_params> -o <output_folder> -l <label> --ngen <num_sequences>
```
where `num_sequences` is the number of sequences to be generated. The output will be saved in fasta format at `output_folder/<label>_samples.fasta`.

If the argument `-d <path_data>` is provided, the script will also compute the Pearson correlation coefficient and the slope between the two-sites correlation matrix of the data and the generated samples.

## Licence
Tish package is oen-sourced under the Apache License 2.0.

## Citation
If you use this package in your research, please cite
> Trinquier, J., Uguzzoni, G., Pagnani, A. et al. Efficient generative modeling of protein sequences using simple autoregressive models. Nat Commun 12, 5800 (2021).

