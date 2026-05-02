# Generate all methods

## Alpaca dataset
python gen_model_answer.py --model-id "llama-2-alpaca-AsFT-reg1" --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --use-lora --adapter-path "../../../experiments/AsFT/finetuned_models/alpaca/AsFT_reg0_top100_clean"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-alpaca-AsFT-reg1
