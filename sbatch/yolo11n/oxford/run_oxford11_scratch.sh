#!/bin/bash
#SBATCH --job-name=Y11n_Oxf_Scr
#SBATCH --partition=ampere            # ΚΡΑΤΑΜΕ ΤΗΝ A100!
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/yolo11n_logs/oxford11_scratch_out_%j.txt
#SBATCH --error=logs/yolo11n_logs/oxford11_scratch_error_%j.txt

echo "=== STARTING JOB: YOLO11-Nano Oxford Pets (From Scratch) ==="

export PYTHONIOENCODING=utf8
export LC_ALL=C.UTF-8

# 1. Φορτώνουμε τα επίσημα modules του ΑΠΘ μαζί με CUDA/cuDNN
module load gcc/13.2.0-i python/3.11 cuda cudnn

# 2. Ενεργοποίηση Env
source /home/l/leandrosk/yolo-env/bin/activate
cd /home/l/leandrosk/

# 3. Libdevice fix (απαραίτητο για το XLA compiler της GPU)
if [ -n "$CUDA_HOME" ]; then
    SYS_LIBDEVICE=$(find $CUDA_HOME -name "libdevice.10.bc" 2>/dev/null | head -n 1)
    if [ -n "$SYS_LIBDEVICE" ]; then
        ln -sf "$SYS_LIBDEVICE" ./libdevice.10.bc
    fi
fi
if [ ! -f "./libdevice.10.bc" ]; then
    ENV_LIBDEVICE=$(find /home/l/leandrosk/yolo-env -name "libdevice.10.bc" | head -n 1)
    if [ -n "$ENV_LIBDEVICE" ]; then
        ln -sf "$ENV_LIBDEVICE" ./libdevice.10.bc
    fi
fi

export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_CPP_MIN_LOG_LEVEL=2

mkdir -p logs/yolo11n_logs

# 4. Έλεγχος GPU
echo "Checking GPU availability..."
python3 -c "import tensorflow as tf; GPUs = tf.config.list_physical_devices('GPU'); import sys; sys.exit(0) if len(GPUs) > 0 else sys.exit(1)"

if [ $? -ne 0 ]; then
    echo "ERROR: TensorFlow cannot see the GPU! Aborting job."
    exit 1
fi
echo "SUCCESS: GPU is online!"

echo "Running Training Script..."
python3 -u server_training/train_oxford_11n.py

echo "=== JOB FINISHED ==="