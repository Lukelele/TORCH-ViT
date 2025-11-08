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
