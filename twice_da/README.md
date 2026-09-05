# TWICE-DA

This repository is the official implementation of **[TWICE-DA: Transformer With Integrated Multi-Scale Convolutional Extractor and Deformable Attention](https://computeroptics.ru/KO/PDF/KO50-1/500114.pdf?ysclid=mtobga4ai3234337)**.

TWICE-DA is designed as an **universal visual feature extractor** for a wide range of computer vision tasks.  
In this repository, we evaluate the effectiveness of the TWICE-DA encoder as a visual backbone for **image classification**.

## Architecture
TWICE-DA is a **lightweight hybrid architecture** that combines the strengths of CNN's and Transformers. The architecture follows a four-stage hierarchical structure, enabling the extraction and processing of visual features at different levels of abstraction.  
An overview of the encoder architecture is presented in the figure below.
<p align="center">
  <img src="assets/twice-da.png" height="700">
</p>

### Key Features
The hybrid nature of TWICE-DA is achieved through a substantial modification of the standard Transformer block, incorporating the following key architectural components:
1. **Multi-Scale Perception Unit (MSPU)** employs multiple parallel convolutional branches to extract local features at different spatial scales. It is placed before the self-attention mechanism to enhance positional information about object locations and introduce stronger convolutional inductive biases into the model.
2. **[Efficient Channel Attention (ECA)](https://arxiv.org/pdf/1910.03151)** dynamically identifies the most informative feature channels, enhancing important representations while suppressing less relevant ones.
3. **[Deformable Multi-Head Attention (DMHA)](https://arxiv.org/pdf/2309.01430)** dynamically samples a limited number of relevant spatial positions for each token using learnable offsets. This allows the model to effectively capture global context while significantly reducing the computational cost compared to vanilla MHSA.
4. **ConvFFNeXt** is a lightweight variant of the feed-forward network (FFN), in which a depthwise convolution is placed before the FFN layers, forming a more computationally efficient ConvNeXt-inspired block. The expansion ratio (e.r.) is reduced from 4 to 2, substantially lowering the computational complexity of the model with only a minor loss in accuracy.
5. **Batch Normalization** is used instead of Layer Normalization. Experimental results demonstrate that the use of Batch Normalization in TWICE-DA improves inference efficiency by approximately 15%, while also providing faster convergence and improved classification accuracy.

### Modification of the DMHA
