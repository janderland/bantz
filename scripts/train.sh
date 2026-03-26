source .venv/bin/activate

mlx_lm.lora \
  --model "$1" \
  --data build/data \
  --adapter-path build/adapters \
  --train \
  --iters "$2" \
  --batch-size "$3" \
  --max-seq-length "$4"
