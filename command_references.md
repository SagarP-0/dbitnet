# HPC AES Differential Experiment: Command Reference

Commands collected from this chat, grouped by task. Run **local PC**
commands in a Linux Mint terminal and **HPC** commands after SSHing into
the cluster.

## 1. Connect to the HPC

**Local PC**

``` bash
ssh sagarp@198.18.0.46
```

## 2. Prepare the teammate's separate workspace

**HPC**

``` bash
mkdir -p ~/pnks
cd ~/pnks
```

Clone the upstream repository into its own directory:

``` bash
git clone https://github.com/CryptAnalystDesigner/differential-neural-cryptanalysis-on-aes.git upstream_repo
```

Verify the required source file:

``` bash
ls ~/pnks/upstream_repo/AES_128_batch.py
```

Inspect the workspace:

``` bash
ls
```

## 3. Transfer the experiment script to the HPC

Run this on the **local PC**, not inside SSH:

``` bash
scp /home/sagar/Documents/MiniProject/pnks/run_diff.py \
  sagarp@198.18.0.46:~/pnks/run_diff.py
```

If the local script has a different filename, replace the source path
accordingly.

## 4. Set up the Python environment

**HPC**

``` bash
conda activate aes-cryptanalysis-gpu
cd ~/pnks
```

Check the TensorFlow version:

``` bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 5. Request a GPU allocation

**HPC login node**

``` bash
salloc -p gpu --qos=normal_3 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=1-00:00:00
```

Start a shell on the allocated compute node:

``` bash
srun --pty bash -l
```

Check GPU visibility and allocation status:

``` bash
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
squeue -u sagarp
```

The allocation ends when its time limit expires or it is relinquished.
Exiting the interactive allocation shell can relinquish it.

## 6. Run the differential experiment

**On the allocated GPU compute node**

``` bash
conda activate aes-cryptanalysis-gpu
cd ~/pnks
```

Run only difference value 128:

``` bash
python run_diff.py \
  --start 128 \
  --end 128 \
  --output_dir "$HOME/pnks/results_diff128"
```

Run values 1 through 10:

``` bash
python run_diff.py \
  --start 1 \
  --end 10 \
  --output_dir "$HOME/pnks/results_1_10"
```

Run values 1 through 128:

``` bash
python run_diff.py \
  --start 1 \
  --end 128 \
  --output_dir "$HOME/pnks/results_1_128"
```

Run one value for debugging (example: 5):

``` bash
python run_diff.py \
  --start 5 \
  --end 5 \
  --output_dir "$HOME/pnks/debug_diff05"
```

The script skips a difference when its result JSON already exists. It
does not resume an interrupted training run within a difference.

## 7. Inspect experiment results

**HPC**

``` bash
ls -lh ~/pnks/results_diff128
cat ~/pnks/results_diff128/b0_0x80.json
```

List result files:

``` bash
ls -lh ~/pnks/results_1_128
```

## 8. Monitor the GPU and SLURM job

**HPC**

``` bash
nvidia-smi
squeue -u sagarp
```

Follow a job's standard output log:

``` bash
tail -f ~/pnks/logs/JOBID.out
```

Replace `JOBID` with the actual job ID.

## 9. Use tmux for a persistent terminal

Start a tmux session:

``` bash
tmux new -s gpu-session
```

Detach without closing the tmux session: press `Ctrl+B`, release, then
press `D`.

Reattach after reconnecting:

``` bash
tmux attach -t gpu-session
```

List tmux sessions:

``` bash
tmux ls
```

**Important:** tmux preserves the terminal process, not the SLURM
allocation. If the allocation is relinquished, cancelled, or reaches its
time limit, the training process on the compute node will be terminated
too.

## 10. Optional: submit the experiment with SLURM

Create the log directory before submitting:

``` bash
mkdir -p ~/pnks/logs
```

Example batch script, saved as `~/pnks/run_diff.slurm`:

``` bash
#!/bin/bash
#SBATCH --job-name=aes-diff
#SBATCH --partition=gpu
#SBATCH --qos=normal_3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/export/home/sagarp/pnks/logs/%j.out
#SBATCH --error=/export/home/sagarp/pnks/logs/%j.err

mkdir -p "$HOME/pnks/logs"

source /export/apps/Anaconda3/2025.12-2/etc/profile.d/conda.sh
conda activate aes-cryptanalysis-gpu

cd "$HOME/pnks"

python run_diff.py \
  --start 128 \
  --end 128 \
  --output_dir "$HOME/pnks/results_diff128"
```

Submit the job:

``` bash
sbatch ~/pnks/run_diff.slurm
```

Check job status:

``` bash
squeue -u sagarp
```

## 11. Transfer results from HPC to the local PC

First, leave the SSH session and run these commands in a **local PC
terminal**.

Create the local destination:

``` bash
mkdir -p /home/sagar/Documents/MiniProject/dbitnet_results
```

Copy the teammate's script:

``` bash
scp sagarp@198.18.0.46:~/pnks/run_diff.py \
  /home/sagar/Documents/MiniProject/pnks/
```

Copy the teammate's results:

``` bash
rsync -av --progress \
  sagarp@198.18.0.46:~/pnks/results/ \
  /home/sagar/Documents/MiniProject/pnks/results/
```

Copy the full DBitNet experiment output directory:

``` bash
rsync -av --progress \
  sagarp@198.18.0.46:~/aes_dbitnet_outputs/ \
  /home/sagar/Documents/MiniProject/dbitnet_results/
```

Check local disk usage and files:

``` bash
du -sh /home/sagar/Documents/MiniProject/dbitnet_results/
ls -lh /home/sagar/Documents/MiniProject/dbitnet_results/
```

`rsync` can be rerun to copy new or changed files without retransferring
everything.

## 12. Useful navigation and inspection commands

**HPC**

``` bash
pwd
ls
ls -la
cd ~/pnks
cd upstream_repo
cd ..
```

Check the experiment output directories:

``` bash
ls -lh ~/aes_dbitnet_outputs/
ls -lh ~/pnks/
```

------------------------------------------------------------------------

## Notes

-   Use `--start` and `--end` with this version of `run_diff.py`; it
    does not use the older `--diff_val` argument.
-   The script expects `AES_128_batch.py` inside
    `~/pnks/upstream_repo/`.
-   Keep the teammate's files under `~/pnks/` and your main project
    outputs under their separate project directory.
-   Do not run local destination paths such as `/home/sagar/...` from
    the HPC. Run transfer commands from the local PC so the destination
    is written to your laptop.
-   Confirm the cluster's current partition, QoS, and wall-time limits
    before requesting a long allocation.
