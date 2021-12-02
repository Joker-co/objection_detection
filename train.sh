export PYTHONPATH=.:$PYTHONPATH

g=$(($2<8?$2:8))
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
    --job-name=debug \
python train.py 
