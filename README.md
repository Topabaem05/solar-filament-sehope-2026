# SE-HOPE native-grid solar filament baseline

Reproducible first-entry baseline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). This is the implemented S1 portion of a broader v0.2 design. It is a small-data baseline, not a completed state-of-the-art model.

## What the model actually does

1. Read each supplied 2048 by 2048 H-alpha grayscale JPEG. Divide intensities by 255, repeat the channel three times, and normalize with mean 0.5 and standard deviation 0.25.
2. Cover every pixel using 25 native-resolution 512 by 512 tiles, at stride 384. Tile locations depend only on image dimensions.
3. Extract frozen VisionHOPE-Tiny features at strides 4, 8, 16 and 32 using the pinned official CUDA backend. There is no persistent memory across images.
4. Apply a width-128 feature pyramid and a three-layer transformer decoder with 64 learned object queries. Attention reads stride-16/32 features; mask logits use stride-4 features. This custom small decoder is not a Mask2Former reproduction.
5. Filter objectness and objectness-times-mask-confidence at 0.5, threshold masks at 0.5, suppress same-tile duplicates at IoU 0.8, and associate overlapping tile instances at overlap IoU 0.5 with at least four shared foreground pixels. Association rejects clusters with conflicting instances from the same tile. Taper-weighted fusion and final IoU-0.8 suppression produce full-size instance masks.
6. Encode each predicted instance as compressed COCO RLE. A zero-detection image contributes zero CSV rows and is explicitly accounted for in the companion manifest. No artificial instances are added.

The inference function receives pixels, model outputs and fixed settings. It never receives annotations, ground-truth boxes, filament spines or test metadata.

## Training and selection

The supplied training data contains 707 distinct images and 1,154 annotator records. The fixed split groups repeated annotations by original image and nearby observations into temporal groups. A bounded registered image-duplicate review was performed; this is not proof against every possible observation correlation.

There are 601 development images across three folds and 106 held-out images. Fold 0 has 400 available training images and 201 validation images. **This first-entry checkpoint was trained on only eight of those 400 training images**, sampled deterministically with seed 42. Initial model screening used two of the 201 validation images. The frozen source records exact filenames and hashes in `evidence/run_inputs.json` and the checkpoint.

Only the query decoder and feature pyramid are trained. AdamW uses learning rate 0.0003, weight decay 0.01 and four accumulated patches per update. Training uses shuffled 16-update blocks of originals, then a uniformly sampled annotator; half of patches are uniformly selected positive grid tiles and half are uniform. These target-dependent crops are training-only. Hungarian matching combines objectness with balanced foreground/background BCE and Dice costs. The loss is object BCE plus five times mask BCE plus five times Dice loss. Unmatched-query object weights are 0.1.

Two controlled runs used 256 and 1,024 optimizer updates, with all other settings fixed. On the same two full validation images their PQ values were 0.061886 and 0.137201. The 1,024-update checkpoint is the candidate. The small screening result is not an estimate of final competition performance. Full-development evaluation, a locked holdout check and submission status are recorded separately in the release evidence; incomplete work is not counted as completed.

## Numerical execution contract

The submitted path uses `official_cuda_v1`, Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4 and an NVIDIA T4 (compute capability 7.5). TF32 and cuDNN benchmarking are disabled; all inference is FP32. The training process and an independent process reproduced the pilot validation CSV byte for byte. Fresh-session reproduction is a separate stage.

The portable CPU implementation and official CUDA kernels did **not** satisfy a previously proposed elementwise `rtol=0.001, atol=0.0001` contract at every element. We do not claim cross-backend equivalence. Source, weight, runtime and implementation hashes are checked, and the checkpoint loader rejects a different backend. Use the official pinned CUDA environment to reproduce this entry.

## Execution status

The [public development notebook](https://www.kaggle.com/code/topabaem/filament-v02-submission-dev-002) is associated with the competition. Its source has been checked against the local notebook cell by cell. Execution and full-development results are still pending; public availability is not an execution-success claim.

## Reproduction files

- `src/filament/`: immutable model, training, full-image inference, RLE and evaluator sources used by the checkpoint.
- `vendor/`: pinned upstream sources, provenance and original MIT notices.
- `weights/decoder.pt`: the trained decoder checkpoint, including configuration and provenance. The frozen backbone is downloaded separately using the public pinned URL and SHA256 in `vendor/sehope_snapshot/weights_manifest.json`.
- `configs/native_s1_official_pilot.json`: the actual 1,024-update configuration.
- `manifests/splits_final_v02.json`: the original-image split; never replace it with a random per-annotation split.
- `requirements.txt`: exact direct runtime dependency versions and the CUDA wheel index.
- `scripts/predict_native.py`: annotation-free prediction from a checked checkpoint.
- `scripts/run_submission_stage.py`: independent replay followed by registered full-development or holdout/test inference.
- `evidence/`: actual training and evaluation reports. Planned improvements are not measured results.

Download competition images through your own authorized Kaggle access. Original images and annotation masks are not redistributed by this repository. `reproduce_development.ipynb` is the complete development replay/evaluation notebook, including isolated setup and the frozen source bundle. The final holdout/test notebook will follow after full-development validation. This candidate release is not yet a completed competition submission.

## Known limitations and further work

The decoder has 1,549,825 parameters. The selected pilot processed its two validation originals in 11.009 seconds on one T4 (environment setup and CUDA compilation excluded), with peak allocated CUDA memory of 546,332,160 bytes during the run. These are small-sample timings.

False positives, fragmentation and overlapping predictions remain material. The public reference evaluator can count multiple IoU-positive pairs, so we also report strict one-to-one PQ. The 1,024-update training score is 0.186436 under the public arithmetic and 0.124765 under strict matching; the two-image validation score is 0.137201 under both. These metrics must not be conflated.

The current checkpoint has no learned native-pixel refiner, stride-8 query attention, topology loss, hard-negative mining, global context branch, ensemble or test-time augmentation. Such additions are research directions, not claims about this entry. Relevant motivations include [Mask2Former](https://arxiv.org/abs/2112.01527), [PointRend](https://arxiv.org/abs/1912.08193), [clDice](https://arxiv.org/abs/2003.07311), and [solar detection followed by segmentation](https://arxiv.org/abs/2402.15407).

## Attribution

The backbone is [VisionHOPE](https://github.com/PSRben/VisionHOPE), commit `98cfb5d6004472a0ba3d212d3cf884393e39c134`. This project builds on the user's SE-HOPE adapter and retains vendor notices. Competition data is [MAGFiLO](https://doi.org/10.1038/s41597-024-03876-y), used under the competition terms. Code is MIT licensed; original upstream notices and dataset terms remain applicable. AI assistance was used to implement and document this pipeline. The participant must understand and be able to explain its methodology as required by the competition rules.
