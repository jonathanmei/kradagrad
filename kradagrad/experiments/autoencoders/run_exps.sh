# PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
# python autoencoders.py --dataset mnist --optimizer adam --epochs 70 

# PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
# python autoencoders.py --dataset faces --optimizer adam --epochs 150

# PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
# python autoencoders.py --dataset curves --optimizer adam --epochs 150

# PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
# python autoencoders.py --dataset mnist --optimizer sgd --epochs 70 

# PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
# python autoencoders.py --dataset faces --optimizer sgd --epochs 150

# PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
# python autoencoders.py --dataset curves --optimizer sgd --epochs 150

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset mnist --optimizer shampoo --epochs 70 \
--block_size 250

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset faces --optimizer shampoo --epochs 150 \
--block_size 500

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset curves --optimizer shampoo --epochs 150 \
--block_size 100

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset mnist --optimizer krad --epochs 70 \
--block_size 250

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset faces --optimizer krad --epochs 150 \
--block_size 500

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset curves --optimizer krad --epochs 150 \
--block_size 100

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset mnist --optimizer kradmm --epochs 70 \
--block_size 250

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset faces --optimizer kradmm --epochs 150 \
--block_size 500

PYTHONPATH=/home/luke.walters/experiments:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
python autoencoders.py --dataset curves --optimizer kradmm --epochs 150 \
--block_size 100
