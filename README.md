# StyTips: Towards High-Quality, Efficient and Controllable Style Transfer via Transformer Filtering Prompts
![](https://github.com/I2-Multimedia-Lab/StyTips/blob/main/network.png)
# Abstract

Image style transfer aims to merge the artistic characteristics of a style image with the spatial structure of a content image, maintaining the content identity and showing the desired style. It has been proven that previous works based on Convolutional Neural Networks (CNNs) lack the ability to capture global features, leading to the content leakage issue. In this work, we propose Transformer fIlter PromptS, dubbed StyTips, a novel approach for style transfer that distills essential content and injects style information via learn prompts. \textcolor{red}{To achieve efficient and high-quality stylization}, we adopt a hierarchical vision Transformer encoder to extract multi-scale features of the content and style images, learning disentangled style features of color and stroke. Inspired by the prompt learning technique, we design a fine-grained style alignment block in the decoder, involving self-attention with learnable filter tokens to distill necessary content information and cross-attention to inject styles. StyTips takes full advantage of the attention mechanism to align the color and stroke with the purified content features in appropriate proportions. Qualitative and quantitative experiments demonstrate that StyTips can prevent content leakage and generate high-quality stylized images. StyTips provides controllable style transfer with diverse ways for users to manipulate styles on fine-grained levels and to determine the strength of style, which has more practical significance to real-world applications than traditional approaches.

# datasets

In the training stage, [MS-COCO](https://cocodataset.org/#download) which is a large-scale dataset for object detection is used as the content dataset, and [WikiArt](https://www.kaggle.com/c/painter-by-numbers) as the style dataset which contains art paintings from different artists.

# Requirments

python == 3.8.13  pytorch == 1.10.0  torchvision == 0.11.0

# Train

run python train.py

# Acknowledgments
The codes of Swin-Transformer encoder are token from (https://github.com/microsoft/Swin-Transformer)
