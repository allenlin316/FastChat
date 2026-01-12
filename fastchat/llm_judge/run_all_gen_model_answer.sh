# Generate all methods
for seed in 20; do

#python gen_model_answer.py --model-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-full-only-finetuning/dolly-7b-full-ft/${seed}-epoch=5" --model-id "Ours-full-only-finetuning"

#python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "Ours-lora-only-finetuning" --use-lora --adapter-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-lora-only-finetuning/dolly-7b-full-ft/${seed}-epoch=10"

#python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "Ours-QLoRA-only-finetuning" --use-qlora --adapter-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-QLoRA-only-finetuning/dolly-7b-full-ft/${seed}-epoch=10"

#python gen_model_answer.py --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --model-id "Ours-SMART-only-finetuning" --use-qlora --adapter-path "../../../experiments/bppo_finetuning/dolly_ours_outliers_normalize_false_length_true_smart_kl/20"

done