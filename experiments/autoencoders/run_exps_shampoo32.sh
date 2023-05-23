
PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders_shampoo32.py --dataset mnist --optimizer shampoo --epochs 70 \
--single --block_size 250

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders_shampoo32.py --dataset curves --optimizer shampoo --epochs 150 \
--single --block_size 100

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH \
python autoencoders_shampoo32.py --dataset faces --optimizer shampoo --epochs 150 \
--single --block_size 500
