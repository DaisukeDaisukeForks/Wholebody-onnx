# Data, pretrained weights, credits, and distribution

This document identifies the sources configured for the production training pipeline. The published model card and training manifest record which sources were actually consumed, their download URLs, selected samples, and preprocessing. A dataset listed as optional below is not a claim that it was used.

## COCO 2017 images and original annotations

Source: [COCO Consortium](https://cocodataset.org/). Terms: [official COCO terms source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). Original COCO annotations are CC BY 4.0: [license](https://creativecommons.org/licenses/by/4.0/). The images retain their individual owners' rights and Flickr license conditions; the annotation license does not relicense the images.
Image distribution: `https://images.cocodataset.org/train2017/{image_id:012d}.jpg` and `https://images.cocodataset.org/val2017/{image_id:012d}.jpg`; the loader can use the corresponding COCO S3 origin. Image IDs, original URLs, Flickr URLs, and license IDs from source metadata are retained in the training manifest. This project distributes trained parameters and metadata; the model publication does not include COCO image copies.
Credit: Tsung-Yi Lin and the COCO Consortium, *Microsoft COCO: Common Objects in Context*, ECCV 2014. [Paper](https://arxiv.org/abs/1405.0312).
Changes: selected person instances are cropped/resized for training; annotation coordinates are converted to the documented crop coordinate system. Official training and validation partitions remain separate.

## COCO-WholeBody v1.0

Official source and terms: [jin-s13/COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody#terms-of-use). Annotation owner: SenseTime Research. The official page identifies the annotations as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and separately limits dataset use to research and noncommercial purposes. These two statements are both retained here. The upstream terms must not be described as unrestricted commercial permission, or silently replaced with a different Creative Commons license.
Configured annotation distribution: [msdkhairi/coco2017 on Hugging Face](https://huggingface.co/datasets/msdkhairi/coco2017/tree/main), specifically `coco_wholebody_train_v1.0.json` and `coco_wholebody_val_v1.0.json`. This is an acquisition mirror, not a new rights holder or a replacement license. Original download alternatives are linked by the official project.
Credit: Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, and Ping Luo, *Whole-Body Human Pose Estimation in the Wild*, ECCV 2020. [Paper](https://arxiv.org/abs/2007.11858).
Changes: body, feet, face, and hand labels are mapped to the documented 133-joint order; invalid or missing labels are excluded from supervision. Person cropping, resizing, and sample selection change the representation, not the ownership of source annotations.

## ASPset-510

Official source: [anibali/aspset-510](https://github.com/anibali/aspset-510). Dataset distribution: [Internet Archive / aspset510](https://archive.org/details/aspset510). License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), confirmed by the official project's acknowledgments and license section.
Providers: La Trobe University and the Australian Institute of Sport. Credit: Aiden Nibali, Joshua Millward, Zhen He, and Stuart Morgan, *ASPset: An Outdoor Sports Pose Video Dataset With 3D Keypoint Annotations*, Image and Vision Computing, 2021. [DOI](https://doi.org/10.1016/j.imavis.2021.104196).
The configured inputs are the official common split metadata, training/validation camera parameters, and training/validation 3D joint archives. Changes: C3D labels are selected by joint name, positions are transformed into camera coordinates, millimeters are converted to meters, the pelvis origin is subtracted, and camera projection produces corresponding 2D training inputs. The pipeline uses those labeled 3D observations for lifting supervision; it does not claim 3D ground truth for all 133 WholeBody joints.

## Torchvision MobileNetV3 Small initialization

Implementation: [pytorch/vision](https://github.com/pytorch/vision). Software license: [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE), copyright Soumith Chintala, 2016, and project contributors. Architecture reference: Andrew Howard et al., *Searching for MobileNetV3*, ICCV 2019. [Paper](https://arxiv.org/abs/1905.02244).
Pretrained initializer: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`. Official distribution: [mobilenet_v3_small-047dcff4.pth](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth). [Official model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
The software's BSD license must not be treated as an unconditional license for every pretrained weight. [Torchvision's pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license) states that model-specific or training-dataset terms may apply. [ImageNet access terms](https://www.image-net.org/download.php) describe noncommercial research and educational use of its database. No ImageNet image archive is downloaded by this pipeline; the official Torchvision initializer supplies the backbone weights. This model release makes no claim that these sources authorize commercial deployment.
Changes: the classification head is discarded and the backbone is connected to newly trained WholeBody localization and body-joint lifting heads. The run metadata records whether this initializer was used.

## Optional candidate excluded from the default training run

CMU Human Foot Keypoint Dataset: [official source and downloads](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/), Carnegie Mellon University; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh; the associated OpenPose paper is linked by the source. Its six 2D foot points are not 3D sole-shape labels. The default run uses the foot labels already present in COCO-WholeBody and does not separately ingest CMU Foot, AIST++, or Human3.6M.

## Project distribution and applicable terms

Development source: [DaisukeDaisuke/Wholebody-onnx](https://github.com/DaisukeDaisuke/Wholebody-onnx).
Intended public model distribution: [RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3](https://huggingface.co/RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3). Intended private checkpoint and gradient transport: [RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3-checkpoints](https://huggingface.co/RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3-checkpoints). These destination names do not assert that model files have already been published. The training workflow creates these new repositories through `HF_TOKEN`, and its output and generated model card identify the actual published revision.
The publication includes this notice, `MODEL_LICENSE.md`, the trained ONNX, model interface metadata, and training provenance. Project source code preserves the repository's existing [Unlicense](https://unlicense.org/) in the root `LICENSE`; it does not override third-party data or pretrained-weight conditions. Trained weights use `MODEL_LICENSE.md`.
