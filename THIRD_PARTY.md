# Third-party sources, licenses, and distribution

This file identifies the data, pretrained parameters, and adapted implementation configured for WholeBody Pose73. The published manifest records the sources actually consumed, file hashes, selected samples, and preprocessing. Dataset terms, implementation licenses, and the publisher's model-release terms are recorded separately.

## COCO 2017 images and original annotations

Source: [COCO Consortium](https://cocodataset.org/). [Official terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). Original annotations are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images retain their owners' rights and individual Flickr conditions.

Distribution: `https://images.cocodataset.org/train2017/{image_id:012d}.jpg` and the corresponding `val2017` path; the loader can use the COCO S3 origin. The manifest retains image IDs, original/Flickr/download URLs, and license metadata. Public model files do not include the source image collection.

Credit: Tsung-Yi Lin and the COCO Consortium, *Microsoft COCO: Common Objects in Context*, ECCV 2014. [Paper](https://arxiv.org/abs/1405.0312). Changes: person cropping, resizing, and coordinate conversion. Official training and validation images remain separate.

## COCO-WholeBody v1.0

Official source and terms: [jin-s13/COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody#terms-of-use), SenseTime Research. The official page identifies annotations as **CC BY 4.0** and separately limits dataset use to **research and noncommercial purposes**. Both statements are retained; this project does not silently replace them with a different Creative Commons license.

Configured distribution: [msdkhairi/coco2017](https://huggingface.co/datasets/msdkhairi/coco2017/tree/main), files `coco_wholebody_train_v1.0.json` and `coco_wholebody_val_v1.0.json`. This is an acquisition mirror, not a new rights holder. Original downloads are linked by the official project.

Credit: Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, and Ping Luo, *Whole-Body Human Pose Estimation in the Wild*, ECCV 2020. [Paper](https://arxiv.org/abs/2007.11858).

Changes: source annotations retain 133-point indexing; supervision selects 12 body joints, 40 finger joints, six foot points, and 15 face points. Hand roots share body wrists. Invalid or absent labels are masked. Facial contour points are not cheek-surface centers; eye corners/lids are not pupil or gaze labels.

Definitions: [official data format](https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md), [MMPose IDs](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py), [iBUG 68-point markup](https://ibug.doc.ic.ac.uk/resources/300-W/). iBUG supplies a definition reference here; no additional 300-W data are downloaded.

## H3WB paired 2D and 3D annotations

Source: [wholebody3d/wholebody3d](https://github.com/wholebody3d/wholebody3d). Annotation license: **MIT**, copyright 2022 H3WB 3D WholeBody Dataset Annotations; [official license](https://github.com/wholebody3d/wholebody3d/blob/main/LICENSE.md), reproduced in [LICENSES/H3WB-MIT.txt](LICENSES/H3WB-MIT.txt).

Official distribution:

- [Recommended NPZ annotations](https://drive.google.com/file/d/1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC/view).
- [Original JSON annotations](https://drive.google.com/file/d/1WVscFZcFIxGi_doifFP3GLTsIbU-hjXO/view).
- [Released 2D-to-3D test annotations](https://drive.google.com/file/d/1xrzfR63pflnw8L9e631xyKe4TwUOOm4k/view).

Credit: Yue Zhu, Nermin Samet, and David Picard, *H3WB: Human3.6M 3D WholeBody Dataset and Benchmark*, ICCV 2023, pp. 20166–20177. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_H3WB_Human3.6M_3D_WholeBody_Dataset_and_Benchmark_ICCV_2023_paper.html). H3WB uses a multi-view reconstruction pipeline; its complete annotations are not uniformly physical-marker measurements.

Changes: select the same 73 COCO-layout points, convert camera coordinates from millimeters to meters, subtract the hip midpoint, normalize corresponding 2D inputs, and apply input noise, missing-point masks, and confidence variation during lifting training. Valid H3WB points provide 3D supervision for fingers, feet, and face as well as the body. The inspected NPZ contains 60,000 camera-frame candidates from S1/S5/S6 for training and 20,000 from S7 for validation. All views and actions of a subject remain in the same split; counts after validity filtering are recorded.

The actual NPZ contains only `metadata` and `train_data`, not supplied bounding boxes. Crop coordinates use the extent of all 133 finite source 2D landmarks, aspect 0.75, and expansion 1.25. Training and inference then share the lifter's hip/RMS normalization. The pinned source SHA-256 is `d423224a9e3c47aa8320687a8fcb58e732aa8d3aabe589e01ce390030d123b1d`.

Human3.6M images remain subject to the [Human3.6M agreement](https://vision.imar.ro/human3.6m/eula.php), separate from the H3WB annotation MIT license. This pipeline uses H3WB coordinate annotations and does not acquire or redistribute Human3.6M RGB images. Human3.6M credit: Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu, *Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments*, IEEE TPAMI 2014; Catalin Ionescu, Fuxin Li, and Cristian Sminchisescu, *Latent Structured Models for Human Pose Estimation*, ICCV 2011.

## ASPset-510

Source: [anibali/aspset-510](https://github.com/anibali/aspset-510). Distribution: [Internet Archive / aspset510](https://archive.org/details/aspset510). License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), as stated by the official project.

Providers: La Trobe University and the Australian Institute of Sport. Credit: Aiden Nibali, Joshua Millward, Zhen He, and Stuart Morgan, *ASPset: An Outdoor Sports Pose Video Dataset With 3D Keypoint Annotations*, Image and Vision Computing, 2021. [DOI](https://doi.org/10.1016/j.imavis.2021.104196).

Inputs: common split metadata, train/validation camera parameters, and train/validation 3D joints. Changes: select C3D labels by name, transform into camera coordinates, convert mm to m, subtract the hip midpoint, and project into 2D. Only the 12 anatomically corresponding body joints supervise this source. Its masks do not invent finger, face, heel, toe, hand-surface, or sole-normal labels.

RGB evaluation additionally acquires the official held-out test clip `1e28/0001/left`, its calibrated camera, 3D joints, and annotated person boxes. The model receives 192 evenly spaced RGB frames cropped using the official boxes; only the matching body12 joints are scored. This is a one-clip, one-camera RGB-to-body3D measurement, not a whole-body RGB-to-3D benchmark or a person-detection evaluation. Exact archive/member URLs, checksums, selected frame IDs, and alignment are recorded in the evaluation manifest. Source RGB frames are not included in the public model distribution.

## DWPose-t / RTMPose-t pretrained pose parameters

Official projects: [MMPose / RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) and [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose). The latter links the [author's model distribution](https://huggingface.co/yzd-v/DWPose), whose [model card](https://huggingface.co/yzd-v/DWPose/blob/main/README.md) explicitly identifies **Apache-2.0**.

Configured OpenMMLab checkpoint:

```text
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth
```

The acquired checkpoint is 22,178,722 bytes. Its pinned SHA-256 is `dcf277bf6ba4563d5ce7b1bede0f78fd5fcd518128b8f38cdf3d74e100addc51`. Its 218 tensor names/shapes match the port's declared architecture; production initialization additionally requires a strict state load. This source inspection is not an inference-accuracy result.

[Official model table](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md) and [tiny configuration](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py). The model was trained for WholeBody pose on COCO-WholeBody and UBody, with upstream pose pretraining recorded by its configuration. This project does not download UBody or AIC to reproduce that earlier training.

Credit: Tao Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, 2023, [paper](https://arxiv.org/abs/2303.07399); Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li, *Effective Whole-body Pose Estimation with Two-stages Distillation*, ICCV workshop 2023, [paper](https://arxiv.org/abs/2307.15880).

Changes: preserve the 256×192 input, select 73 final-convolution channels, reuse compatible shared MLP/GAU/SimCC weights, fine-tune the resulting smaller joint context, and attach the confidence-aware full-body lifter. The original 133-point teacher and initial 73-point student provide separate validation baselines. Removing joints changes GAU attention; neither identical outputs nor unchanged upstream AP are asserted. Actual download hashes and initialization coverage are recorded by the run.

## Adapted implementation

`pose_architecture.py` ports CSPNeXt/RTMCC components from [MMPose](https://github.com/open-mmlab/mmpose), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMCV](https://github.com/open-mmlab/mmcv) to pure PyTorch. These components retain **Apache-2.0** and OpenMMLab attribution. DWPose attribution is also retained.

Full license: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Copyright and modification notice: [LICENSES/DWPose-NOTICE.txt](LICENSES/DWPose-NOTICE.txt). The original orchestration, data preparation, and export code keeps the source repository's [Unlicense](https://github.com/DaisukeDaisuke/Wholebody-onnx/blob/main/LICENSE); it does not relicense the Apache components. Model distributions also include this source-code license as `CODE_LICENSE`.

The port removes registry/runtime dependencies while retaining compatible layer definitions and parameter names. New compact outputs and lifting behavior are project modifications, not upstream performance claims.

## Optional mixed-precision conversion

The exporter optionally uses Microsoft's `onnxconverter-common==1.16.0`, distributed through [PyPI](https://pypi.org/project/onnxconverter-common/1.16.0/) under the [MIT License](https://github.com/microsoft/onnxconverter-common/blob/v1.16.0/LICENSE), copyright Microsoft Corporation. [Source](https://github.com/microsoft/onnxconverter-common/tree/v1.16.0) and [official conversion API](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html). The conversion package is a build dependency; its license does not replace the pretrained-model or dataset conditions.

Changes to the exported artifact: convert only the extracted backbone/final-convolution segment to FP16 and rejoin it with the original FP32 input normalization and FP32 GAU/SimCC/lifter/geometry. Public input/output types remain FP32. The optional file is published only after it meets the same declared held-out quality criteria. Exact precision boundaries, file hashes, sizes, evaluation provider, and reused-metric scope are recorded. CPU evaluation can promote unsupported FP16 operators to FP32 and is not evidence of native GPU behavior or performance.

## Training transport

The transport uses [`huggingface_hub==1.33.0`](https://github.com/huggingface/huggingface_hub/tree/v1.33.0), distributed under [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/v1.33.0/LICENSE). Official references: [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets) and the [versioned Python Bucket API](https://huggingface.co/docs/huggingface_hub/v1.33.0/guides/buckets). Buckets provide mutable object storage for training state and intermediate files. The client license does not replace the model/data licenses or the hosting service's terms.

The revised training transport writes binary objects before their control JSON. Complete checkpoints use immutable per-update paths; a last-written pointer identifies the checkpoint and its SHA-256. Readers verify the referenced bytes. Gradients and per-round weight exchange use the private bucket; the private model repository receives phase backups and final candidates. This removes per-round Git commit creation. Service retry/reset delays remain respected.

The execution layout changes from two physical workers with32 accumulation batches to eight with8, while retaining two logical data shards and their cursors. Global batches remain512 2D examples and4,160 3D examples. The saved common step17 and its AdamW state come from the completed two-worker updates of [run41444692224](https://github.com/DaisukeDaisuke/Wholebody-onnx/actions/runs/41444692224); at this revision, the eight-worker/bucket continuation has not yet run. This infrastructure change does not alter the pretrained-model or dataset attribution.

## Distribution

Source: [DaisukeDaisuke/Wholebody-onnx](https://github.com/DaisukeDaisuke/Wholebody-onnx).

- Intended public model: [RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3](https://huggingface.co/RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3).
- Private phase backups and final candidates: [RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-checkpoints](https://huggingface.co/RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-checkpoints).
- New private training exchange bucket: [RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-exchange](https://huggingface.co/buckets/RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-exchange).

GitHub Actions creates these repositories and the private bucket under RabbitDaisuke. Their configured URLs do not imply completed training or publication. Published files include this notice, [MODEL_LICENSE.md](MODEL_LICENSE.md), the full `LICENSES/` directory, model/interface files, and actual run provenance. Upstream Apache rights and dataset conditions remain distinct from the publisher's additional model-release grant.
