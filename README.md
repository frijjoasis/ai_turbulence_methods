Code used to produce the results in the paper *Comparison of Generative Learning Methods as Turbulence Surrogates* (former: *Comparison of Generative Learning Methods for Turbulence Modeling*). The paper can be found on ArXiv: [https://arxiv.org/abs/2411.16417](https://arxiv.org/abs/2411.16417). 
Code for the DDPM model is in review at [https://github.com/NVIDIA/modulus](https://github.com/NVIDIA/modulus).

## Datasets
The experiments are performed on three datasets obtained by LES and PIV and are available on Zenodo:
* LES (Flow around a cylinder): https://zenodo.org/records/13820259 (Save *Turbulence_AI.zip* to a desired directory)
    * Contains 100,000 grayscale images of resolution 1,000 $\times$ 600.
* PIV (Flow behind array of seven cylinders): https://doi.org/10.5281/zenodo.16794036 (Save *CylinderArrays.mat* to a desired directory)
    * Contains images of nine different test cases (2V2H, 2V4H, 2V8H, 4V2H, 4V4H, 4V8H, 6V2H, 6V4H, 6V8H) saved in .mat-file format.
    * Each test case has 3,000 images of resolution 138 $\times$ 231.
    * The test cases considered in our paper are 2V8H and 4V2H.

## Repository Structure
The repo consists of four folders containing the scripts for the respective generative models (DCGAN, DDPM, VAE) and the computation of the evaluations:

```plaintext
ai_turbulence_methods/
├── dcgan_karman/
│   ├── requirements.txt
│   ├── options/
|   |   ├── inference_options.py
|   |   ├── train_options.py
│   ├── dcgan_inference_piv.py
│   ├── dcgan_inference_les.py
│   ├── dcgan_train_piv.py
│   ├── dcgan_train_les.py
│   └── utils.py
├── diffusion_karman/
│   ├── requirements.txt
│   ├── config.json
│   ├── dataset.py
│   ├── distribute.py
│   ├── main.py
│   ├── model.py
│   ├── params.py
│   ├── sample.py
│   ├── trainer.py
│   └── util.py
├── vae_karman/
│   ├── requirements.txt
│   ├── params.json
│   ├── distribute.py
│   ├── model.py
│   ├── params.py
│   ├── util.py
│   ├── vae-sample.py
│   └── vae-train.py
├── evaluations/
│   ├── evaluation_les.py
│   ├── evaluation_piv.py
│   └── postprocessing_ddpm_vae.py
└── README.md
```

## How to - Training the models

The considered generative models each have their own training routines, as described below. The actual parameter settings for all models on both datasets are provided in the paper.

### DCGAN
To train the DCGAN, run `python dcgan_train_les.py` or `python dcgan_train_piv.py` when using the LES or PIV data, respectively.

The code is written in a user-friendly way, meaning that the dataset directory, checkpoint path, and training parameters can be set either in the options script `train_options.py` or directly in the command line, e.g.

`python dcgan_train_piv.py --dataset_dir \your\data\directory --checkpoint_dir \your\path\to\save\ckpt --outer_key 0 --num_epochs 2000 --resize_img 128`

All customizable parameters are explained in the options script. For the PIV dataset, it is also necessary to set a key to choose the desired test case. To use 2V8H, set `outer_key` to 3, and to use 4V2H, set `outer_key` to 1.

If not set differently, the outputs of the training (image progress, models, and options log) will be saved in the working directory in the folder `train_output`.

### DDPM

### VAE

## How to - Using the trained models

After training, the saved checkpoints can be used to generate new samples. This process is referred to as inference, which can be performed as explained below.

### DCGAN 
Run `python dcgan_inference_les.py` or `python dcgan_inference_piv.py` when the model has been trained on the LES or PIV data, respectively.

As in the case of DCGAN training, the checkpoint paths, generator models, and other parameters can be customized within the options script `inference_options.py` or directly in the command line. All customizable parameters are also explained in the options script.

If not set differently, the outputs of the inference will be saved in the working directory in the folder `inference_output/experiment_name`. For models trained on the LES dataset, the outputs are a desired number of grayscale images. For models trained on the PIV dataset, the outputs consist of grayscale images for the two channels *u* and *v*, respectively, and a `.pt` file called `generated`, which contains the exact values of the generated samples.

### DDPM

### VAE

## Evaluations
After generating new samples with the trained models, it’s time to perform the physics-based evaluations introduced in our paper. To ensure fairness, the number of compared images should be the same for the LES/PIV, DCGAN-, DDPM-, or VAE-generated data.  

For the LES dataset, the grayscale images are used for the evaluation. In the case of the PIV dataset, the exact values from the `.pt` files are loaded. To run the scripts, adjust the file paths in the `__main__` part below the comment  
`# Set the data paths`. Be cautious that the samples of all models (DCGAN, DDPM, VAE) and the training data (LES/PIV) are in the same domain, as the images are normalized during training.  

Attention must be paid in the case of the PIV dataset: The DCGAN-generated samples are denormalized to the original domain already at inference time. For the DDPM and VAE models, this has to be done through postprocessing. Run `postprocessing_ddpm_vae.py` with your customized setup (search for the comment `# CUSTOMIZE`) and use the resulting denormalized samples, which are in the original domain, for the evaluation.

