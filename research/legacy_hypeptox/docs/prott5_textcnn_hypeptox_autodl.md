# ProtT5/TextCNN HyPepTox-Fuse AutoDL Commands

Run from the GHXTox project root on AutoDL.

## 1. Install ProtT5 Dependencies

```bash
pip install -e ".[prott5]"
```

If editable extras are unavailable in the current environment:

```bash
pip install transformers sentencepiece protobuf
```

## 2. Prepare ProtT5 Embeddings

Set `MODEL_DIR` to the local ProtT5 directory on the server. The directory should contain the HuggingFace files for `prot_t5_xl_half_uniref50-enc`.

```bash
MODEL_DIR=/root/autodl-tmp/models/prot_t5_xl_half_uniref50-enc
CACHE_DIR=/root/autodl-tmp/GHXTox/data/plm/prot_t5_xl_half_uniref50-enc
```

Generate ProtT5 features for the no-leak split and independent test:

```bash
python -m ghxtox.plm_embed \
  --input data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2.pt \
  --output data/external/hypeptox_fuse/processed/train_fit_chemical_sites_prott5.pt \
  --model prot_t5_xl_half_uniref50-enc \
  --model-path "$MODEL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size 32

python -m ghxtox.plm_embed \
  --input data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
  --output data/external/hypeptox_fuse/processed/train_val_chemical_sites_prott5.pt \
  --model prot_t5_xl_half_uniref50-enc \
  --model-path "$MODEL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size 32

python -m ghxtox.plm_embed \
  --input data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
  --output data/external/hypeptox_fuse/processed/test_chemical_sites_prott5.pt \
  --model prot_t5_xl_half_uniref50-enc \
  --model-path "$MODEL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size 32
```

## 3. Augment Only The ProtT5 Train-Fit Split

```bash
python -m ghxtox.augment_processed \
  --input data/external/hypeptox_fuse/processed/train_fit_chemical_sites_prott5.pt \
  --output data/external/hypeptox_fuse/processed/train_fit_chemical_sites_prott5_augmix.pt \
  --copies-per-record 1.0 \
  --boundary-fraction 0.6 \
  --k-neighbors 8 \
  --lambda-min 0.25 \
  --lambda-max 0.75 \
  --noise-std 0.01 \
  --seed 42
```

## 4. Train The ProtT5/TextCNN Expert

```bash
python -m ghxtox.train \
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_prott5_augmix.pt \
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_prott5.pt \
  --config configs/hypeptox_fuse_prott5_textcnn_augmix_noleak.json \
  --output-dir runs/hypeptox_fuse_prott5_textcnn_augmix_noleak \
  --device cuda
```

Additional seeds:

```bash
python -m ghxtox.train --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_prott5_augmix.pt --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_prott5.pt --config configs/hypeptox_fuse_prott5_textcnn_augmix_noleak_seed123.json --output-dir runs/hypeptox_fuse_prott5_textcnn_augmix_noleak_seed123 --device cuda
python -m ghxtox.train --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_prott5_augmix.pt --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_prott5.pt --config configs/hypeptox_fuse_prott5_textcnn_augmix_noleak_seed2025.json --output-dir runs/hypeptox_fuse_prott5_textcnn_augmix_noleak_seed2025 --device cuda
```

## 5. Evaluate And Ensemble

```bash
python -m ghxtox.evaluate \
  --checkpoint runs/hypeptox_fuse_prott5_textcnn_augmix_noleak/best_model.pt \
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_prott5.pt \
  --output runs/hypeptox_fuse_prott5_textcnn_augmix_noleak/test_metrics.json \
  --predictions runs/hypeptox_fuse_prott5_textcnn_augmix_noleak/test_predictions.csv \
  --device cuda
```

To combine the ProtT5/TextCNN expert with the current 7-model ESM2 ensemble, use one processed file per checkpoint.

First select the threshold on validation data:

```bash
python -m ghxtox.ensemble_evaluate \
  --processed-per-checkpoint \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/train_val_chemical_sites_prott5.pt \
  --checkpoints \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_3d_v2_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_prott5_textcnn_augmix_noleak/best_model.pt \
  --optimize-threshold-metric mcc \
  --output runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_prott5_8model/val_threshold_metrics.json \
  --device cuda
```

Then evaluate the unchanged independent test with the validation-selected threshold:

```bash
THRESHOLD_FROM_VAL=0.386

python -m ghxtox.ensemble_evaluate \
  --processed-per-checkpoint \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
    data/external/hypeptox_fuse/processed/test_chemical_sites_prott5.pt \
  --checkpoints \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_3d_v2_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_prott5_textcnn_augmix_noleak/best_model.pt \
  --threshold "$THRESHOLD_FROM_VAL" \
  --output runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_prott5_8model/test_metrics.json \
  --predictions runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_prott5_8model/test_predictions.csv \
  --device cuda
```
