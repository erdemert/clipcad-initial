#!/bin/bash
#SBATCH --job-name=cadclip
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=100G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Usage:
#   sbatch train.slurm                    # plain CLIP loss (default)
#   sbatch train.slurm cad_guided         # cad_guided loss, beta=1.0 (default)
#   sbatch train.slurm cad_guided 0.5     # cad_guided loss, beta=0.5
#
# Positional args override any already-exported CAD_CLIPPER_LOSS_TYPE /
# CAD_CLIPPER_CAD_GUIDED_BETA, which override the hardcoded defaults below — so
# `sbatch --export=ALL,CAD_CLIPPER_CAD_GUIDED_BETA=0.3 train.slurm cad_guided` also works
# (env var wins over the default, positional arg wins over the env var).
#
# train.py's BATCH_SIZE defaults to 512 (sized for clip) but reads CAD_CLIPPER_BATCH_SIZE if
# set — lower it for cad_guided, whose per-step cost is O(batch_size^2):
# `sbatch --export=ALL,CAD_CLIPPER_BATCH_SIZE=32 train.slurm cad_guided`
export CAD_CLIPPER_LOSS_TYPE="${1:-${CAD_CLIPPER_LOSS_TYPE:-clip}}"
export CAD_CLIPPER_CAD_GUIDED_BETA="${2:-${CAD_CLIPPER_CAD_GUIDED_BETA:-1.0}}"

cd /mnt/DTX_AI_lab/erdem_erturk_0/clipcad_multiview/clipcad-initial || exit 1

# The #SBATCH --output above can't see $1/$2 -- Slurm parses #SBATCH lines before this script
# ever runs, so %x/%j alone can't reflect which config a run used. Redirecting here instead,
# now that the config is known, means the log FILENAME itself encodes it (e.g.
# logs/cadclip_cad_guided_beta0.5_123456.out), so `ls logs/` tells runs apart without opening
# each one. Anything printed before this line (or a Slurm-level failure before the script
# even starts) still lands in the original slurm-%x-%j.out as a fallback.
mkdir -p logs
LOG_FILE="logs/cadclip_${CAD_CLIPPER_LOSS_TYPE}_beta${CAD_CLIPPER_CAD_GUIDED_BETA}_${SLURM_JOB_ID}.out"
exec > "$LOG_FILE" 2>&1

echo "loss_type=$CAD_CLIPPER_LOSS_TYPE  cad_guided_beta=$CAD_CLIPPER_CAD_GUIDED_BETA  job_id=$SLURM_JOB_ID"

source /mnt/amax6_drive/erdem_erturk_0/CAD_Project/cad-project/bin/activate

python -u train.py
