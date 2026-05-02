# Generate all methods
for seed in 42; do
## Dolly dataset
# python gen_model_answer.py --model-id "llama-2-dolly-SFT-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-lora-final_min5_cand10/dolly-7b-lora-ft/42_merged/"

# python gen_model_answer.py --model-id "llama-2-dolly-Bianchi-safety-k10-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Bianchi-safety-augmentation_k10/dolly-7b-lora-ft/42_merged/"

# python gen_model_answer.py --model-id "llama-2-dolly-lambda0.5-mu10-seed${seed}" --model-path "../../../experiments/bppo_finetuning/Dolly/Llama-2-7B-Chat-fp16/dolly_lora_ft_smart_kl_lambda0.5_mu10/42_merged/"

## Alpaca dataset
# python gen_model_answer.py --model-id "llama-2-alpaca-SFT-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Alpaca/Ours-lora-final_min5_cand10/alpaca-7b-lora-ft/42_merged/"

# python gen_model_answer.py --model-id "llama-2-alpaca-Bianchi-safety-k10-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Alpaca/Bianchi-safety-augmentation_k10/alpaca-7b-lora-ft/42_merged/"

# python gen_model_answer.py --model-id "llama-2-dolly-lambda0.5-mu10-seed${seed}" --model-path "../../../experiments/bppo_finetuning/Alpaca/Llama-2-7B-Chat-fp16/alpaca_lora_ft_smart_kl_lambda0.5_mu10/42_merged/"


## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-dolly-SFT-seed${seed} llama-2-dolly-Bianchi-safety-k10-seed${seed} llama-2-alpaca-SFT-seed${seed} llama-2-alpaca-Bianchi-safety-k10-seed${seed}

done
