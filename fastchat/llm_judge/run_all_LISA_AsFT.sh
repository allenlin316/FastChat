# Generate all methods
#for seed in 106; do

echo "running seed 20 on LISA & AsFT methods"
seed=20
## Dolly dataset
python gen_model_answer.py \
    --model-id "llama-2-dolly-LISA-seed${seed}" \
    --model-path /workspace/models/Llama-2-7b-hf \
    --use-lora \
    --adapter-path "../../../experiments/Lisa/ckpt/Llama-2-7b-hf_sft/" \
    --adapter-path2 "../../../experiments/Lisa/ckpt/dolly/Llama-2-7b-hf_lisa_f_1_0_100_10_10_10000/20/"

python gen_model_answer.py \
    --model-id "llama-2-dolly-AsFT-seed${seed}" \
    --model-path "/workspace/models/Llama-2-7B-Chat-fp16" \
    --use-lora \
    --adapter-path "../../../experiments/AsFT/finetuned_models/dolly/Llama-2-7B-Chat-fp16/dolly_AsFT_reg1_top100_clean/20/"

## Alpaca dataset
python gen_model_answer.py \
    --model-id "llama-2-alpaca-LISA-seed${seed}" \
    --model-path /workspace/models/Llama-2-7b-hf \
    --use-lora \
    --adapter-path "../../../experiments/Lisa/ckpt/Llama-2-7b-hf_sft/" \
    --adapter-path2 "../../../experiments/Lisa/ckpt/alpaca/Llama-2-7b-hf_lisa_f_1_0_100_10_10_10000/20/"

python gen_model_answer.py \
    --model-id "llama-2-alpaca-AsFT-seed${seed}" \
    --model-path "/workspace/models/Llama-2-7B-Chat-fp16" \
    --use-lora \
    --adapter-path  "../../../experiments/AsFT/finetuned_models/alpaca/Llama-2-7B-Chat-fp16/alpaca_AsFT_reg1_top100_clean/20/"

echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-dolly-LISA-seed${seed} llama-2-dolly-AsFT-seed${seed} llama-2-alpaca-LISA-seed${seed} llama-2-alpaca-AsFT-seed${seed}

echo "running seed 106 only on LISA"
seed=106
## Dolly dataset
python gen_model_answer.py \
    --model-id "llama-2-dolly-LISA-seed${seed}" \
    --model-path /workspace/models/Llama-2-7b-hf \
    --use-lora \
    --adapter-path "../../../experiments/Lisa/ckpt/Llama-2-7b-hf_sft/" \
    --adapter-path2 "../../../experiments/Lisa/ckpt/dolly/Llama-2-7b-hf_lisa_f_1_0_100_10_10_10000/106/"

## Alpaca dataset
python gen_model_answer.py \
    --model-id "llama-2-alpaca-LISA-seed${seed}" \
    --model-path /workspace/models/Llama-2-7b-hf \
    --use-lora \
    --adapter-path "../../../experiments/Lisa/ckpt/Llama-2-7b-hf_sft/" \
    --adapter-path2 "../../../experiments/Lisa/ckpt/alpaca/Llama-2-7b-hf_lisa_f_1_0_100_10_10_10000/106/"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-dolly-LISA-seed${seed} llama-2-alpaca-LISA-seed${seed}

#done