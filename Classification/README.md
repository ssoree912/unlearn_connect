# Saliency-Unlearning for Classification
This is the official repository for Saliency Unlearning for Clasification. The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.


## Requirements
```bash
pip install -r requirements.txt
```

## Scripts
1. Get the origin model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {epochs for training} --lr {learning rate for training} --save_dir {file to save the orgin model}
    ```

    A simple example for ResNet-18 on CIFAR-10.
    ```bash
    python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182
    ```

2. Generate Saliency Map
    ```bash
    python generate_mask.py --save_dir ${saliency_map_path} --model_path ${origin_model_path} --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs 1
    ```

3. Unlearn
    *  SalUn
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --num_indexes_to_replace ${forgetting data amount} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
    ```

    A simple example for ResNet-18 on CIFAR-10 to unlearn 10% data.
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --num_indexes_to_replace 4500 --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path mask/with_0.5.pt
    ```

    To compute UA, we need to subtract the forget accuracy from 100 in the evaluation results. As for MIA, it corresponds to multiplying SVC_MIA_forget_efficacy['confidence'] by 100 in the evaluation results. For a detailed clarification on MIA, please refer to Appendix C.3 at the following link: https://arxiv.org/abs/2304.04934.


    * Retrain
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn retrain --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * FT
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * GA
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn GA --num_indexes_to_replace 4500 --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * IU
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn wfisher --num_indexes_to_replace ${forgetting data amount} --alpha ${alpha}
    ```

    * l1-sparse
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT_prune --num_indexes_to_replace ${forgetting data amount} --alpha ${alpha} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

## Fixed-forget LMC workflow
The repository now supports fixed forget indexes, separate data/unlearning seeds, and intermediate checkpoint saving for checkpoint-wise linear mode connectivity experiments.

1. Create a fixed forget set once.
    ```bash
    python make_forget_indices.py \
      --arch resnet18 \
      --dataset cifar10 \
      --num_indexes_to_replace 4500 \
      --forget_seed 1 \
      --output_path ../artifacts/forget_indices.npy
    ```

2. Reuse that forget set for mask generation and unlearning.
    ```bash
    python generate_mask.py \
      --save_dir runs/mask_fixed \
      --model_path ${origin_model_path} \
      --forget_seed 1 \
      --forget_index_path ../artifacts/forget_indices.npy \
      --unlearn_seed 1 \
      --unlearn_epochs 1
    ```

    ```bash
    python main_random.py \
      --unlearn RL \
      --unlearn_epochs 10 \
      --unlearn_lr 0.013 \
      --model_path ${origin_model_path} \
      --save_dir runs/salun_A \
      --mask_path runs/mask_fixed/with_0.5.pt \
      --forget_seed 1 \
      --forget_index_path ../artifacts/forget_indices.npy \
      --unlearn_seed 11 \
      --checkpoint_epochs 0,1,3,5,10
    ```

3. Evaluate saved checkpoints into `endpoint_metrics.csv`.
    ```bash
    python evaluate_checkpoints.py \
      --arch resnet18 \
      --dataset cifar10 \
      --run_dir runs/salun_A \
      --unlearn RL \
      --forget_seed 1 \
      --forget_index_path ../artifacts/forget_indices.npy \
      --include_final_checkpoint
    ```

4. Measure linear interpolation between two runs.
    ```bash
    python interpolate_checkpoints.py \
      --arch resnet18 \
      --dataset cifar10 \
      --run_a_dir runs/salun_A \
      --run_b_dir runs/salun_B \
      --curve_epochs 0,1,3,5,10 \
      --forget_seed 1 \
      --forget_index_path ../artifacts/forget_indices.npy \
      --output_dir ../artifacts/interpolation \
      --retrain_metrics_path runs/retrain/endpoint_metrics.csv
    ```

## Nested ratio sweep
For ratio sweeps such as 10, 20, 30, 40, 50 percent random forgetting, create one shared permutation and use nested prefixes so that each larger forget set contains the smaller one.

1. Generate nested forget index files under `<RUNS_DIR>/<ratio>/forget_indices.npy`.
    ```bash
    python make_nested_forget_indices.py \
      --arch resnet18 \
      --dataset cifar10 \
      --forget_seed 1 \
      --percentages 10,20,30,40,50 \
      --output_root runs \
      --permutation_output_path runs/forget_permutation.npy
    ```

2. Run the two-stage ratio-aware sweep. The script keeps the baseline outside the ratio folders, tunes SalUn first with a single seed under `<RUNS_DIR>/_tuning/<ratio>/keep_<keep>/lr_<lr>/`, selects the best config for each ratio, and only then runs the final `salun_A`, `salun_B`, and interpolation jobs.
    ```bash
    bash run_nested_ratio_sweep.sh
    ```
    To keep a parameter setting completely separate from existing checkpoints, set `EXPERIMENT_NAME` or `EXPERIMENT_ROOT`. For example:
    ```bash
    EXPERIMENT_NAME=keepgrid_v1_skipmia bash run_nested_ratio_sweep.sh
    ```
    This writes outputs under `experiments/keepgrid_v1_skipmia/runs/` while still reusing the default baseline checkpoint at `runs/baseline/0checkpoint.pth.tar` unless `BASE_CKPT` is overridden.

    Per ratio, the sweep writes:
    - `<RUNS_DIR>/<ratio>/best_salun.env`
    - `<RUNS_DIR>/<ratio>/best_salun_leaderboard.csv`
    - `<RUNS_DIR>/<ratio>/salun_A/endpoint_metrics.csv`
    - `<RUNS_DIR>/<ratio>/salun_B/endpoint_metrics.csv`
    - `<RUNS_DIR>/<ratio>/interpolation/barrier_summary.csv`

    After the sweep finishes, aggregate CSVs are refreshed under `<SUMMARY_DIR>/`:
    - `<SUMMARY_DIR>/all_endpoint_metrics.csv`
    - `<SUMMARY_DIR>/all_barrier_summary.csv`
    - `<SUMMARY_DIR>/all_retrain_gap_summary.csv`

3. The current default tuning mode is `retrain_oracle`. It compares each candidate against the retrain oracle for that forgetting ratio using `ua,acc_retain,acc_test,mia`.
    ```bash
    FORGET_SEED=1 RATIOS_CSV=20 bash run_nested_ratio_sweep.sh
    ```
    This is the recommended first step: tune `20%` only, inspect `best_salun_leaderboard.csv`, and only then move to the next ratio. The default `20%` oracle-tuning sweep now uses eight explicit candidates centered on the current oracle-following region instead of a full Cartesian grid.

4. To switch to paper-target selection instead of oracle-based selection:
    ```bash
    SELECTOR_MODE=paper_target TUNING_SKIP_MIA=0 SELECTOR_SCORE_COLS=ua,acc_retain,acc_test,mia bash run_nested_ratio_sweep.sh
    ```

5. By default the sweep now stops after tuning and best-config selection. To run the final two-seed SalUn A/B checkpoints and interpolation for a ratio that already has a selected config:
    ```bash
    FORGET_SEED=1 RATIOS_CSV=20 RUN_RETRAIN=0 RUN_TUNING=0 RUN_SALUN=1 RUN_INTERPOLATION=1 bash run_nested_ratio_sweep.sh
    ```

6. To add only `FT` and `GA` final endpoints after the SalUn sweep:
    ```bash
    RUN_RETRAIN=0 RUN_SALUN=0 RUN_INTERPOLATION=0 RUN_FT=1 RUN_GA=1 bash run_nested_ratio_sweep.sh
    ```

7. To regenerate CSV artifacts only from existing checkpoints, without rerunning training or unlearning:
    ```bash
    bash extract_ratio_csvs.sh
    ```
    This script rewrites per-run `endpoint_metrics.csv`, per-ratio interpolation CSVs, and the aggregate CSVs under `<SUMMARY_DIR>/`. It skips missing runs automatically.

In this implementation, `generate_mask.py` saves `with_<x>.pt` where `x` is the mask keep ratio, not the paper's sparsity label. The current sweep defaults are:
- SalUn keep grids: `10->0.2 0.3 0.4 0.5 0.6 0.7`, `20->0.45 0.48 0.50 0.55 0.60`, `30->0.33 0.34 0.35 0.36`, `40->0.1 0.2 0.3 0.4`, `50->0.1 0.2 0.3 0.4`
- SalUn lr grids: `10->0.005 0.008 0.013 0.02 0.03`, `20->0.0175 0.018 0.019`, `30->0.0165 0.017 0.0175`, `40->0.001 0.002 0.003 0.005 0.008`, `50->0.0005 0.001 0.002 0.003 0.005`
- SalUn epoch grids: `10->10`, `20->15`, `30->10`, `40->10`, `50->10`
- Default `20%` explicit SalUn candidate set: `(0.50,0.0175,15)`, `(0.50,0.018,15)`, `(0.55,0.0175,15)`, `(0.55,0.018,15)`, `(0.45,0.018,15)`, `(0.48,0.018,15)`, `(0.60,0.0175,15)`, `(0.50,0.019,15)`
- Default `30%` explicit SalUn candidate set: `(0.33,0.0165,10)`, `(0.34,0.0165,10)`, `(0.34,0.017,10)`, `(0.35,0.0165,10)`, `(0.35,0.017,10)`, `(0.34,0.0175,10)`, `(0.36,0.017,10)`, `(0.36,0.0175,10)`
- Tuning checkpoint epochs default to `6,8,10,12,15`, with `20%` overridden to `10,12,15` and `30%` overridden to `6,8,10`
- For unlearning epochs `10/12/15`, the default learning-rate decay milestones are `5,8 / 6,10 / 8,12`
- Oracle selector weights default to `ua=2.5,mia=1.0,acc_test=0.7,acc_retain=0.7`
- Oracle selector constraints currently applied at `20%`: `acc_retain >= 98.3`, `acc_test >= 92.8`
- Optional paper SalUn targets: `10->(2.85,99.62,93.93,14.39)`, `20->(3.73,98.61,92.75,13.18)`, `30->(6.22,95.91,90.72,14.11)`, `40->(6.86,95.01,89.76,15.15)`, `50->(7.75,94.28,89.29,16.99)` for `(UA, RA, TA, MIA)`
- FT lr centers: `10->0.01`, `20->0.005`, `30->0.003`, `40->0.002`, `50->0.001`
- GA lr centers: `10->3e-5`, `20->1e-5`, `30->3e-6`, `40->1e-6`, `50->1e-6`
- Retrain stays on the full oracle schedule: `182` epochs with lr `0.1`
- GA epochs default to `5`; FT epochs default to `10`

The sweep script can be configured through environment variables such as `BASE_CKPT`, `FORGET_SEED`, `RATIOS_CSV`, `UNLEARN_SEED_TUNE`, `UNLEARN_SEED_A`, `UNLEARN_SEED_B`, `CKPT_EPOCHS`, `TRAIN_BASELINE`, `RUN_TUNING`, `RUN_SALUN`, `RUN_INTERPOLATION`, `RUN_FT`, `RUN_GA`, `SELECTOR_MODE`, `TUNING_SKIP_MIA`, `SELECTOR_SCORE_COLS`, `SUMMARY_DIR`, and `RUN_AGGREGATION`.
