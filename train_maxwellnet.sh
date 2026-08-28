#!/bin/bash
#SBATCH --job-name=hnn
#SBATCH --account=pc_alsrixs
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fliou@lbl.gov

module load python/3.12

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH=$(pwd):$PYTHONPATH

chmod u+w "$SLURM_SUBMIT_DIR"
mkdir -p "$SLURM_SUBMIT_DIR/$SLURM_ARRAY_JOB_ID"
mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$SLURM_SUBMIT_DIR/$SLURM_ARRAY_JOB_ID/"

# train_maxwellnet.py always reads ./specs_maxwell.json (no --config flag) --
# unlike the multi-config array-job template, there's one spec per submit
# dir, so SLURM_ARRAY_TASK_ID isn't used to pick a config/seed here.
poetry run python train_maxwellnet.py
