# Generate all methods
for seed in 20; do

## Dolly Dataset
# python gen_model_answer.py --model-path "../../../experiments/1.1_harmful_scores/Dolly/Full-benign/dolly-7b-full-ft/20-epoch=5" --model-id "llama-2-full-benign"

# python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "llama-2-dolly-lora-final_min5_cand10" --use-lora --adapter-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-lora-final_min5_cand10/dolly-7b-lora-ft/${seed}"

# python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "llama-2-dolly-QLoRA-final_min5_cand10" --use-qlora --adapter-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-QLoRA-final_min5_cand10/dolly-7b-QLoRA-ft/${seed}"

# python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "llama-2-dolly-SMART-LoRA-kl-mu1" --use-lora --adapter-path "../../../experiments/bppo_finetuning/dolly_final_LoRA_FT_smart_kl_mu1/${seed}"

# python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "llama-2-dolly-SMART-LoRA-kl-mu0" --use-lora --adapter-path "../../../experiments/bppo_finetuning/dolly_final_LoRA_FT_smart_kl_mu0/${seed}"

# ## Alpaca dataset
python gen_model_answer.py --model-id llama-2-alpaca-full-benign --model-path ../../../experiments/1.1_harmful_scores/Alpaca/Full-Benign/alpaca-7b-full-ft/20

# python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "llama-2-alpaca-lora-final_min5_cand10" --use-lora --adapter-path "../../../experiments/1.1_harmful_scores/Alpaca/Ours-lora-final_min5_cand10/alpaca-7b-lora-ft/20"

# python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "llama-2-alpaca-QLoRA-final_min5_cand10" --use-qlora --adapter-path "../../../experiments/1.1_harmful_scores/Alpaca/Ours-QLoRA-final_min5_cand10/alpaca-7b-QLoRA-ft/20"

echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-alpaca-full-benign

done