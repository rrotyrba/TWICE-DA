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

**Key features:**  
1. Instead of the typical **Patchify Stem**, which relies on a single convolutional layer to partition the input into patches, TWICE-DA employs a **Convolutional Stem** consisting of two consecutive convolutional layers with a stride of 2. This design enables a more gradual reduction of the feature map resolution while preserving more important visual details, which is particularly beneficial for **dense prediction tasks**.
