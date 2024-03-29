# StyTips:Style Transfer via Transformer Filtering Prompts

# Abstruct

Image style transfer aims to transfer artistic characteristics from a style image to a content image, maintaining the identity of content but with color and stroke of the style image.
It has been proven that previous works based on Convolutional Neural Networks (CNNs) lack the ability to capture global features, leading to the issue of content leakage. In this paper,
we propose Transformer fIlter PromptS, dubbed StyTips, a novel approach to realize style transfer that distills content and injects style information via prompts. The encoder implemented by a hierarchical vision Transformer will extract multi-scale features of content and style images, learning fine-grained style features of color and stroke, which can further extend to multiple style fusion. Inspired by prompt tuning, we design a style alignment module in the decoder, applying self-attention to distill necessary content information under a guide of filtering prompts and cross-attention to inject styles, so that the color and stroke can be aligned with the purified content features in appropriate proportions. Qualitative and quantitative experiments demonstrate that StyTips can prevent content leakage and generate high-quality images with more flexibility and controllability. Moreover, StyTips provides diverse ways for users to manipulate styles on fine-grained levels and determine the strength of style, which has more practical significance to real-world applications than traditional approaches.

# datasets

In the training stage, [MS-COCO](https://cocodataset.org/#download) which is a large-scale dataset for object detection is used as the content dataset, and [WikiArt](https://www.kaggle.com/c/painter-by-numbers) as the style dataset which contains art paintings from different artists.

# Requirments

python == 3.8.13  pytorch == 1.10.0  torchvision == 0.11.0

# Train

run python train.py

# Acknowledgments
The codes of Swin-Transformer encoder are token from (https://github.com/microsoft/Swin-Transformer)
