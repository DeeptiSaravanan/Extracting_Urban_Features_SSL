# Extracting_Urban_Features_SSL

This work aims to study the urban features that affect the crime rate of an area. Such insights would help in better planning of urban infrastructure in the future to tackle various global issues. In our study, we treat the extraction of such urban features from satellite images as an object detection and classification problem. Given the advancements in the field of self-supervised learning (SSL) and the huge availability of unlabeled satellite data, two self-supervised models, Deformable DETRreg and Masked Auto Encoder were tested and compared to detect urban features in the input satellite images. The models were evaluated on two levels. Mean Average Precision (MAP) was used to evaluate the ability to detect urban features. To evaluate whether the detected urban features are relevant, the classification performance of the feature maps of satellite images from the SSL models into high and low crime rates is compared against a baseline classifier.

## Procedure to run the scripts:

### Swav pretraining:

```
cd def_detr/swav_pretrain
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0

python3 setup.py install --cuda_ext
python3 -c 'import apex; from apex.parallel import LARC' # should run and return nothing
python3 -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)' # should run and return apex.parallel.optimized_sync_batchnorm

cd ..

python3 -m torch.distributed.launch --nproc_per_node=1 main_swav.py \
--data_path /unlabeled \
--epochs 15 \
--base_lr 0.6 \
--final_lr 0.006 \
--warmup_epochs 0 \
--batch_size 64 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--checkpoint_freq 1 \
--workers 2 \
```

### Deformable DETR pretraining:

```
cd def_detr/detr_pretrain/models/ops
sh ./make.sh
cd ../..
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_top30_coco.sh --batch_size 8 --epochs 10 --num_workers 2
```

### Deformable DETR finetuning:

```
cd def_detr/detr_finetune/models/ops
sh ./make.sh
cd ../..
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_full_coco.sh --batch_size 2 --epochs 50 --num_workers 2
```

### Deformable DETR prediction:

```
cd def_detr/detr_prediction
python3 evaluate.py
```

### MAE pretraining:

### MAE finetuning:

### MAE prediction:

```
cd mae/mae_prediction
python3 imageCheck.py
```
