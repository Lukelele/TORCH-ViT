# TORCH ViT 
Desnoising Vision Transformer for CERN LHCb TORCH detector.

## Models

This project includes the following models, located in the `Models/` directory:

### 1\. Convolutional Autoencoder

  * **File:** `Models/convolutional_autoencoder.py`
  * **Class:** `ConvolutionAutoencoder`
  * **Description:** A standard convolutional autoencoder consisting of an `encoder` and `decoder` stack. The encoder uses `Conv2d` layers to reduce dimensionality, and the decoder uses `ConvTranspose2d` layers to reconstruct the original input.
      * **Encoder:** 3-layer `torch.nn.Sequential` block with `Conv2d`, `BatchNorm2d`, and `ReLU`.
      * **Decoder:** 3-layer `torch.nn.Sequential` block with `ConvTranspose2d`, `BatchNorm2d`, and `ReLU`.

### 2\. Hybrid Denoising Transformer

  * **File:** `Models/hybrid_denoising_transformer.py`
  * **Description:** This file implements a hybrid architecture that combines a CNN encoder/decoder with a Transformer encoder block for feature processing. It also includes a custom `LearnablePositionalEncoding` module.
  * **Models:**
      * **`HybridTransformerBase`**: The base model with `embed_dim=64`, `num_heads=4`, and `num_layers=4`.
      * **`HybridTransformerTiny`**: A minimal version with `embed_dim=16`, `num_heads=1`, and `num_layers=1`.
      * **`HybridTransformerSmall`**: A small version with `embed_dim=32`, `num_heads=2`, and `num_layers=2`.
      * **`HybridTransformerLarge`**: A large version with `embed_dim=96`, `num_heads=8`, and `num_layers=8`.
  * **Architecture (all sizes):**
    1.  **CNN Encoder:** A `Sequential` block with `Conv2d`, `BatchNorm2d`, `GELU`, and `MaxPool2d` layers.
    2.  **Positional Encoding:** Features are flattened and `LearnablePositionalEncoding` is added.
    3.  **Transformer:** A standard `torch.nn.TransformerEncoder` processes the sequences.
    4.  **Feature Fusion:** A residual connection adds the original CNN features to the transformer output.
    5.  **CNN Decoder:** A `Sequential` block with `ConvTranspose2d` layers upsamples the features back to the original image size.

## Example Usage
```python
import torch
from Models.hybrid_denoising_transformer import HybridTransformerBase

# Generating Monte-carlo Data
train_dataset = TorchData.TORCHData(x=X, y=Y, num_data=NUM_DATA,
                                    signal_count=(SIGNAL_COUNT_MIN, SIGNAL_COUNT_MAX), signal_select_mode=SIGNAL_SELECT_MODE,
                                    noise_density=NOISE_DENSITY, t_offset=T_OFFSET,
                                    blur_level=BLUR_LEVEL, dispersion_level=DISPERSION_LEVEL, mode=MODE)
train_dataloader = train_dataset.dataloader(batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TorchData.TORCHData(x=X, y=Y, num_data=1024,
                                   signal_count=(SIGNAL_COUNT_MIN, SIGNAL_COUNT_MAX), signal_select_mode=SIGNAL_SELECT_MODE,
                                   noise_density=NOISE_DENSITY, t_offset=T_OFFSET,
                                   blur_level=BLUR_LEVEL, dispersion_level=DISPERSION_LEVEL, mode=MODE)
test_dataloader = test_dataset.dataloader(batch_size=1, shuffle=False)



# Example for Vision Transformer
model = HybridTransformerBase()
loss_function = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCH):
    for x, y in train_dataloader:
        optimiser.zero_grad()

        outputs = model(x)
        loss = loss_function(outputs, y)
  
        loss.backward()
        optimiser.step()
```

## Abstract 
The transformer has become the standard for natural language processing, being the core architecture that powers large language models. Recent studies have demonstrated its capabilities in image processing, however, its use has not yet been explored in the context of signal denoising in particle physics. We present a novel hybrid architecture combining convolutional neural networks and the transformer, which is able to denoise simulated data for the LHCb TORCH detector, achieving a signal retention ratio of 0.994$\pm$0.022 and a noise removal ratio of 0.901$\pm$0.033, significantly outperforming traditional architectures based on convolutional neural networks alone.

# Background
The Large Hadron Collider beauty (LHCb) experiment is a specialised detector system of the Large Hadron Collider at CERN, primarily designed to study the physics of Charge Conjugation Parity (CP) violations. Charge Conjugation (C) refers to the mathematical operation that transforms the particle into its anti-particle, while Parity (P) represents a reflection in the spatial coordinates through the origin. The combined CP symmetry was initially thought to be conserved in nature, however in 1964, Cronin and Fitch discovered this symmetry could be violated in weak interactions. Understanding CP violation is crucial as it constitutes one of the three Sakharov conditions necessary to explain the observed matter-antimatter asymmetry in our universe. Without this asymmetry, the early universe would have contained equal amounts of matter and antimatter, resulting in complete annihilation.

<img width="710" height="517" alt="16d135ef65394ad1e2fd006a38c344aa81dfead0" src="https://github.com/user-attachments/assets/cd22f621-2e97-4af3-bb5a-b7eb04465307" />

The LHCb detector is a single-arm forward spectrometer of approximately 20 meters in length, particles enter from the left through the Vertex Locator (VELO) to the right. The detection process begins at the interaction point with the VELO identifying collision vertices. Particles then traverse past a magnet, allowing for momentum measurements through track curvature. The Ring-Imaging Cherenkov (RICH) detectors identify particle types by measuring Cherenkov radiation patterns. Downstream, calorimeters measure particle energies, while dedicated muon chambers detect and count muons.

As part of the LHCb Upgrade II, a newly proposed TORCH (Time of Internally Reflected Cherenkov Light) detector would be installed upstream of the RICH2 detector to enhance the particle identification capabilities, especially at the low momentum ranges. TORCH operates on the Time of Flight (ToF) information, which is the time taken for the particle to travel from the fixed distance from the interaction point at VELO to TORCH. ToF is used in conjunction with the particle momentum to calculate the mass of the particle and thus its type.

TORCH contains an array of photodetectors that detect any Cherenkov photons produced by the incoming charged particle and record the photon Time of Arrival (ToA), which is essential in calculating the time of flight, the details are described in Appendix A. Crucially, the determination of time of flight from time of arrival relies on the reconstruction of each path taken by the photon, which is computationally intensive. Uncorrelated photons constitute around 80\% of all detections, which means a significant portion of compute power is wasted on reconstructing the path of noise photons, which are then discarded. The aim of this project is to develop an algorithm to discard the noise photons before reconstruction, which significantly reduces its computational cost. This algorithm must rely only on the detector positions and time of arrival, without the photon path information.

Machine learning and deep learning techniques have seen a surge in popularity over the past decade, driven by the rapid increase in the large scale parallel GPU compute infrastructure. They have demonstrated to excel in many applications within particle physics, such as the implementation of neural networks in detector trigger systems in the CMS experiment. In this project, we aim to develop neural network algorithms for the purpose of signal denoising. We implement and evaluate 2 architectures, an autoencoder based on convolutional neural networks, which has traditionally shown capabilities in image denoising applications; as well as a novel hybrid transformer architecture adapted from the vision transformer \cite{vit} for signal denoising.

## Results
### Hybrid Transformer Denoising
![5b002dd8e860963becbe2b3dd5ba55eb932a9baa](https://github.com/user-attachments/assets/954682a1-165f-4237-aca7-16ea8c83a992)

### Results Comparison Betwenen Different Model Sizes
<img width="660" height="225" alt="image" src="https://github.com/user-attachments/assets/260fc162-c662-4a50-9f5f-d0fbaee5900b" />

