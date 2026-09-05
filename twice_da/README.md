# TWICE-DA

This repository is the official implementation of **[TWICE-DA: Transformer With Integrated Multi-Scale Convolutional Extractor and Deformable Attention](https://computeroptics.ru/KO/PDF/KO50-1/500114.pdf?ysclid=mtobga4ai3234337)**.

TWICE-DA is designed as a **universal visual feature extractor** for a wide range of computer vision tasks.  
In this repository, we evaluate the effectiveness of the TWICE-DA encoder as a visual backbone for **image classification**.

## Architecture
TWICE-DA is a **lightweight hybrid architecture** that combines the strengths of CNNs and Transformers. The architecture follows a four-stage hierarchical structure, enabling the extraction and processing of visual features at different levels of abstraction.  
An overview of the encoder architecture is presented in the figure below.
<p align="center">
  <img src="assets/twice-da.png" height="700">
</p>

### Key Components
The hybrid nature of TWICE-DA is achieved through a substantial modification of the standard Transformer block, incorporating the following key architectural components:
1. **Multi-Scale Perception Unit (MSPU)** employs multiple parallel convolutional branches to extract local features at different spatial scales. It is placed before the self-attention mechanism to enhance positional information about object locations and introduce stronger convolutional inductive biases into the model.
2. **[Efficient Channel Attention (ECA)](https://arxiv.org/pdf/1910.03151)** dynamically identifies the most informative feature channels, enhancing important representations while suppressing less relevant ones.
3. **[Deformable Multi-Head Attention (DMHA)](https://arxiv.org/pdf/2309.01430)** dynamically samples a limited number of relevant spatial positions for each token using learnable offsets. This allows the model to effectively capture global context while significantly reducing the computational cost compared to standard MHSA.
4. **ConvFFNeXt** is a lightweight variant of the feed-forward network (FFN), in which a depthwise convolution is placed before the FFN layers, forming a more computationally efficient ConvNeXt-inspired block. The expansion ratio (e.r.) is reduced from 4 to 2, substantially lowering the computational complexity of the model with only a minor loss in accuracy.
5. **Batch Normalization** is used instead of Layer Normalization. Experimental results demonstrate that the use of Batch Normalization in TWICE-DA improves inference efficiency by approximately 15%, while also providing faster convergence and improved classification accuracy.

### Modification of the DMHA
We also propose a modification of the standard DMHA by introducing a **Multi-Scale Offset Generator**.  
The standard **Offset Generator (OG)** used in DMHA is implemented as a simple convolutional neural network, as illustrated in the figure below. However, this design has several limitations. In particular, the use of convolutions with a fixed kernel size limits the ability of the offset generator to effectively capture information at different levels of spatial detail. Moreover, processing large feature maps with such convolutions can result in increased computational costs.

To address these limitations, we introduce the **Multi-Scale Offset Generator (MSOG)**, which performs multi-scale feature processing through **pre-aggregation of features**. This enables the generation of more informative and accurate spatial offsets while maintaining **linear computational complexity with respect to the number of input feature-map channels**.
<p align="center">
  <img src="assets/dmha and generators.png" height="700">
</p>

## Model Configuration
In this work, we consider the **TWICE-DA (T)** Tiny variant of the proposed architecture.

The hyperparameters and implementation details of the TWICE-DA-T architecture are presented in the table below.

| Stage | Output shape | Params |
| :--- | :---: | :--- |
| **Stage 1** | $\frac{H}{4} \times \frac{W}{4}$ | $D = 3$, $G = 1$,<br>$C = 64$, $k = [3, 7, 21]$,<br>$R = 8$, $h = [9, 15]$,<br>$H = 2$, $E = 2$. |
| **Stage 2** | $\frac{H}{8} \times \frac{W}{8}$ | $D = 3$, $G = 2$,<br>$C = 128$, $k = [3, 7, 15]$,<br>$R = 4$, $h = [5, 11]$,<br>$H = 4$, $E = 2$. |
| **Stage 3** | $\frac{H}{16} \times \frac{W}{16}$ | $D = 6$, $G = 4$,<br>$C = 256$, $k = [3, 7, 11]$,<br>$R = 2$, $h = [3, 7]$,<br>$H = 8$, $E = 2$. |
| **Stage 4** | $\frac{H}{32} \times \frac{W}{32}$ | $D = 3$, $G = 8$,<br>$C = 512$, $k = [3, 5, 7]$,<br>$R = 1$, $h = [3, 5]$,<br>$H = 16$, $E = 2$. |
