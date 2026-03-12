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
      --forget_seed 2 \
      --output_path ../artifacts/forget_indices.npy
    ```

2. Reuse that forget set for mask generation and unlearning.
    ```bash
    python generate_mask.py \
      --save_dir runs/mask_fixed \
      --model_path ${origin_model_path} \
      --forget_seed 2 \
      --forget_index_path ../artifacts/forget_indices.npy \
      --unlearn_seed 2 \
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
      --forget_seed 2 \
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
      --forget_seed 2 \
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
      --forget_seed 2 \
      --forget_index_path ../artifacts/forget_indices.npy \
      --output_dir ../artifacts/interpolation \
      --retrain_metrics_path runs/retrain/endpoint_metrics.csv
    ```
