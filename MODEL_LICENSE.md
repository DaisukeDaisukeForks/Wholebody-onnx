# WholeBody Pose73 Model Release Terms 1.1

Copyright 2026 DaisukeDaisuke, for the publisher's own contributions.

## Upstream parameters and adapted implementation

The pose initializer is the official DWPose-t / RTMPose-t WholeBody model. DWPose's [author model card](https://huggingface.co/yzd-v/DWPose/blob/main/README.md) identifies Apache-2.0. The adapted CSPNeXt/RTMCC implementation and upstream pretrained material retain their Apache-2.0 terms, copyright notices, and permissions.

The full license is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt), with attribution and project modifications in [LICENSES/DWPose-NOTICE.txt](LICENSES/DWPose-NOTICE.txt). This document does not withdraw or narrow rights already granted by those upstream licensors.

## Additional grant for publisher-owned contributions

To the extent the publisher holds the necessary rights in newly trained contributions and added model components, permission is granted to download, use, study, reproduce, adapt, and redistribute those contributions for noncommercial research and educational purposes. Retain this document, `THIRD_PARTY.md`, the applicable files in `LICENSES/`, and source attribution. State modifications in redistributed adaptations and preserve these conditions for the publisher's licensed contributions.

This additional grant does not replace upstream Apache-2.0, the H3WB annotation MIT license, or any dataset's own conditions. It supplies no additional commercial-use permission for publisher-owned fine-tuning contributions. No endorsement by upstream authors or institutions is implied.

## Training data and scope

The configured model selects 73 COCO-WholeBody 2D landmarks, uses paired H3WB annotations to supervise their 3D coordinates, and adds ASPset-510 supervision for the 12 matching body joints. Invalid or unavailable labels are masked. The input and output definitions, actual training sources, and preprocessing are recorded in the published schema and manifest.

[THIRD_PARTY.md](THIRD_PARTY.md) lists source owners, distribution URLs, credits, transformations, and conditions. COCO-WholeBody identifies its annotations as CC BY 4.0 and separately specifies research/noncommercial dataset use. COCO images retain individual conditions. ASPset-510 is CC0. H3WB annotations carry the [MIT license](LICENSES/H3WB-MIT.txt); Human3.6M image conditions are separate, and this pipeline does not acquire those RGB images.

These statements do not assert that every dataset license automatically becomes a copyright license for all resulting parameters. The publisher grants only its own rights and does not relicense upstream data, images, or parameters.

## Source code and distribution

The project's original code preserves the source repository's [Unlicense](https://github.com/DaisukeDaisuke/Wholebody-onnx/blob/main/LICENSE), also included as `CODE_LICENSE` in model distributions. Apache-covered adaptations are separately identified and are not placed under that public-domain dedication.

Development source: [DaisukeDaisuke/Wholebody-onnx](https://github.com/DaisukeDaisuke/Wholebody-onnx). Intended public model: [RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3](https://huggingface.co/RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3). Private training state: [RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-checkpoints](https://huggingface.co/RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-checkpoints).

GitHub Actions creates the new destinations. Their names are not evidence of completed training. Publication records identify the actual model revision, quality-gate result, and training provenance. Distributions must include the applicable license and attribution files.

## Warranty

Publisher-owned contributions are provided as available, without warranties of accuracy, fitness, or noninfringement, and without liability to the extent permitted by applicable law. Upstream warranty and liability clauses remain in effect for their covered material. Reported metrics apply only to the stated model, data, and evaluation input conditions; they are not guarantees of production GPU usage or unseen-scene accuracy.
