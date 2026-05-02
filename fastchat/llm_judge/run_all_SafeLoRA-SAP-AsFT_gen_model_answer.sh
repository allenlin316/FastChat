# Generate all methods
for seed in 106; do

## Dolly dataset
# python gen_model_answer.py --model-id "llama-2-dolly-SafeLoRA-threshold0.6-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/SafeLoRA-dolly/dolly-7b-lora-safelora/106_merged/"

# python gen_model_answer.py --model-id "llama-2-dolly-SAP-seed${seed}" --model-path "../../../experiments/SAP/SAPcode/ckpts/Llama-2-7B-Chat-fp16/sap_model_dolly_seed106/106_merged/"

python gen_model_answer.py --model-id "llama-2-dolly-AsFT-reg1-seed${seed}" --model-path "../../../experiments/AsFT/finetuned_models/dolly/Llama-2-7B-Chat-fp16/dolly_AsFT_reg1_top100_clean/106_merged"

## Alpaca dataset
# python gen_model_answer.py --model-id "llama-2-alpaca-SafeLoRA-threshold0.6-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Alpaca/SafeLoRA-alpaca/alpaca-7b-lora-safelora/106_merged/"

# python gen_model_answer.py --model-id "llama-2-alpaca-SAP-seed${seed}" --model-path "../../../experiments/SAP/SAPcode/ckpts/Llama-2-7B-Chat-fp16/sap_model_alpaca_seed106/106_merged/"

python gen_model_answer.py --model-id "llama-2-alpaca-AsFT-reg1-seed${seed}" --model-path "../../../experiments/AsFT/finetuned_models/alpaca/Llama-2-7B-Chat-fp16/alpaca_AsFT_reg1_top100_clean/106_merged"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-dolly-SafeLoRA-threshold0.6-seed${seed} llama-2-dolly-SAP-seed${seed} llama-2-alpaca-SafeLoRA-threshold0.6-seed${seed} llama-2-alpaca-SAP-seed${seed} llama-2-dolly-AsFT-reg1-seed${seed} llama-2-alpaca-AsFT-reg1-seed${seed}

done