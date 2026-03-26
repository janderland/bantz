source .venv/bin/activate

mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --data build/data \
  --adapter-path build/adapters \
  --train \
  --iters 1000 \
  --batch-size 1 \
  --max-seq-length 650
