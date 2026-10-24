# Satellite Cloud and Fog Removal with GANs

A Python project for restoring satellite images affected by clouds or fog. It uses a pretrained SPANet generator and provides a local web interface for trying the included models with a sample image or an image of your own.

## What the project does

- Restores an input satellite image using one of the included pretrained models.
- Displays the input, restored image, and model attention map in a browser.
- Lets you upload an image or use the bundled sample.
- Lets you download the restored image.
- Can run inference on a CPU; a CUDA-enabled GPU is optional.

The generator is SPANet, an attention-based image restoration network. The training code also includes a conditional discriminator: it judges cloudy/clean image pairs, while the generator learns to produce realistic restorations. Generator training combines adversarial, L1 reconstruction, and attention losses.

## Requirements

- Python 3.9 or newer
- The pretrained model files included in `pretrained_models/`
- A CUDA-capable PyTorch installation only if you want GPU inference

## Run the web interface

Open a terminal in the project folder and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the app:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Select a model, choose **Use the included sample** or upload a PNG, JPG, or TIFF image, and click **Remove clouds / fog**. Use **Download restored image** to save the restored result.

### Models

- **RICE2 (cloud removal)** — `pretrained_models/RICE2/gen_model_epoch_200.pth`
- **RICE1 (fog removal)** — `pretrained_models/RICE1/gen_model_epoch_200.pth`

The app uses the CPU by default. Select **Use CUDA GPU** in the sidebar if a compatible GPU and CUDA-enabled PyTorch are available.

## Run inference from the command line

The standalone demo saves a comparison image with the input, restoration, and attention map side by side:

```powershell
python demo.py --test_filepath "readme_images\test_0000.png" --pretrained "pretrained_models\RICE2\gen_model_epoch_200.pth" --output "results\demo.png"
```

Replace the sample path with the path to your own image. Use the RICE1 checkpoint for fog removal:

```powershell
python demo.py --test_filepath "path\to\satellite.png" --pretrained "pretrained_models\RICE1\gen_model_epoch_200.pth" --output "results\demo.png"
```

Inference runs on the CPU unless `--cuda` is supplied and CUDA is available.

## Repository layout

```text
app.py                 Streamlit web interface
demo.py                Single-image command-line inference
train.py               GAN training loop
eval.py                Model evaluation
data_manager.py        Dataset loading
models/gen/SPANet.py   Attention-based generator
models/dis/dis.py      Conditional discriminator
pretrained_models/     Included RICE1 and RICE2 generator weights
readme_images/         Bundled demo images
data/RICE_DATASET/     Dataset directory structure
```

## Training data

The dataset directories in this repository contain placeholder files, not the satellite image dataset. Therefore, pretrained inference works without additional data, but training and dataset-based evaluation require the actual RICE dataset. Configure the dataset paths and training parameters in `config.yml` after placing the dataset in the expected structure.

## Notes

This project demonstrates image restoration with pretrained weights. Restored images are model predictions and should not be treated as verified or authoritative satellite observations.

## Credits

The SPANet architecture, GAN training code, and pretrained RICE model weights are based on the original project: [Cloud and Fog Removal with GANs](https://github.com/rahulrajpv/Cloud-and-Fog-removal-System-using-Generative-Adversarial-Networks-Satellite-Image-Processing). This repository adds a Streamlit interface and setup documentation around that work. Please retain this attribution when redistributing the project.
