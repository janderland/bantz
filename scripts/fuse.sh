source .venv/bin/activate
mlx_lm.fuse --model "$1" --adapter-path build/adapters --save-path build/model

jq '.tokenizer_class = "PreTrainedTokenizerFast"' build/model/tokenizer_config.json > build/model/tokenizer_config.tmp.json &&\
  mv build/model/tokenizer_config.tmp.json build/model/tokenizer_config.json
