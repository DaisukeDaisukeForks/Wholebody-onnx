# WholeBody Research Model License 1.0

Copyright 2026 DaisukeDaisuke, for the publisher's own contributions.

## Scope and permission

This license applies to the trained checkpoint parameters and ONNX model files produced by this project. To the extent the publisher holds the necessary rights, permission is granted to download, use, study, reproduce, adapt, and redistribute these model files for noncommercial research and educational purposes, subject to retaining this license, `THIRD_PARTY.md`, and source attribution. Redistributed adaptations must state the changes and preserve these conditions for the publisher's licensed contributions. No endorsement by a dataset author, institution, or upstream software project is implied.

## Third-party material

The model is trained using COCO-WholeBody labels and ASPset-510 3D labels, and may include a Torchvision MobileNetV3 Small ImageNet initializer as recorded in the training manifest. [THIRD_PARTY.md](THIRD_PARTY.md) gives their owners, distribution locations, license links, and transformations. COCO-WholeBody's official page specifies research/noncommercial dataset use while identifying its annotations as CC BY 4.0. ASPset-510 is CC0. COCO images retain their individual image conditions. Torchvision distinguishes its BSD software license from potentially separate pretrained-model conditions.
This document licenses only rights the publisher can grant. It does not relicense upstream datasets, images, or pretrained parameters, resolve the legal treatment of learned weights in every jurisdiction, or assert that a dataset license automatically becomes a copyright license for all resulting parameters. In particular, this release does not grant commercial-use rights. Any rights needed beyond this release must be obtained from the applicable rights holders.

## Source code and distribution

The project's training and export source code preserves the existing Unlicense in the separate root `LICENSE` of [DaisukeDaisuke/Wholebody-onnx](https://github.com/DaisukeDaisuke/Wholebody-onnx). Its source-code license does not change these model conditions. The intended public model destination is [RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3](https://huggingface.co/RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3), with private training state in the separately created `RabbitDaisuke/Wholebody-Parallel-20261102-a6fde3-checkpoints` repository. The configured destination is not a claim of completed publication; the generated model card records the actual published revision.

## Warranty

The publisher provides the model as available, without a warranty of accuracy, fitness, or noninfringement, and disclaims liability to the extent permitted by applicable law. Published metrics describe the stated run and evaluation data only.
