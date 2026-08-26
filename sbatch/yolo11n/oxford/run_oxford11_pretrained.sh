#!/bin/bash
#SBATCH --job-name=Y11n_Oxf_Pre
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=06:00:00               # 6 ώρες (Είναι υπεραρκετές για Transfer Learning)
#SBATCH --output=logs/yolo11n_logs/oxford11_pretrained_out_%j.txt
#SBATCH --error=logs/yolo11n_logs/oxford11_pretrained_error_%j.txt

echo "============================================================"
echo "🚀 ΕΚΚΙΝΗΣΗ JOB: YOLO11-Nano Oxford Pets (Transfer Learning)"
echo "============================================================"

# Φόρτωση βιβλιοθηκών ΑΠΘ
module load gcc/13.2.0-i python/3.11 cuda cudnn

# Ενεργοποίηση Python Environment
echo "🔄 Ενεργοποίηση Python Environment..."
source /home/l/leandrosk/yolo-env/bin/activate
cd /home/l/leandrosk/

# Μικρο-διορθώσεις για A100 (libdevice)
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

# Δημιουργία φακέλου logs αν δεν υπάρχει
mkdir -p logs/yolo11n_logs

# ======================================================
# ΔΙΑΚΟΠΤΗΣ ΑΣΦΑΛΕΙΑΣ GPU
# ======================================================
echo "🔍 Έλεγχος διαθεσιμότητας GPU..."
python3 -c "import tensorflow as tf; GPUs = tf.config.list_physical_devices('GPU'); import sys; sys.exit(0) if len(GPUs) > 0 else sys.exit(1)"

if [ $? -ne 0 ]; then
    echo "❌ ΣΦΑΛΜΑ: Το TensorFlow δεν βλέπει την A100! Το job διακόπτεται."
    exit 1
fi
echo "✅ Η GPU είναι ενεργή."
# ======================================================

echo "⏳ Ξεκινάει η εκπαίδευση με Transfer Learning..."
python3 -u server_training/train_oxford_11n_pretrained.py

echo "✅ ΤΕΛΟΣ JOB!"