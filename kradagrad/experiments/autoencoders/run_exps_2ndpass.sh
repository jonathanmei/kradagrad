
PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset mnist --optimizer sgd --epochs 70 \
--tag "asha"

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset faces --optimizer sgd --epochs 150 \
--tag "asha"

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset curves --optimizer sgd --epochs 150 \
--tag "asha"

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset mnist --optimizer kradmm --epochs 70 \
--block_size 250 --tag "iter" --iterative

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset curves --optimizer kradmm --epochs 150 \
--block_size 100 --tag "iter" --iterative

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset mnist --optimizer krad --epochs 70 \
--block_size 250 --tag "iter" --iterative

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset curves --optimizer krad --epochs 150 \
--block_size 100 --tag "iter" --iterative

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset faces --optimizer kradmm --epochs 150 \
--block_size 500 --tag "iter" --iterative

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders.py --dataset faces --optimizer krad --epochs 150 \
--block_size 500 --tag "iter" --iterative
