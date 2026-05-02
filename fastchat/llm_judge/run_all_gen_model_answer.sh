# Generate all methods

## Dolly dataset
# SFT (w/o defense)
# python gen_model_answer.py --model-id "Ministral-3-dolly" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-Full-final_min5_cand10/dolly-Ministral-8b-full-ft/20/"
# SafeInstr (Bianchi)
python gen_model_answer.py --model-id "Ministral-3-dolly-SafeInstr-k10" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Bianchi-safety-augmentation_k10/dolly-Ministral-8b-full-ft/20/"
# SAP
# python gen_model_answer.py --model-id "Ministral-3-dolly-SAP" --model-path "/workspace/models/Ministral-3-8B-Instruct-textonly" --use-lora --adapter-path "../../../experiments/SAP/SAPcode/checkpoints/Ministral-3-8B-Instruct-textonly/sap_model_dolly/"

# SafeLoRA
# python gen_model_answer.py --model-id "Ministral-3-dolly-SafeLoRA-threshold0.9" --model-path "/workspace/models/Ministral-3-8B-Instruct-textonly" --use-lora --adapter-path "../../../experiments/1.1_harmful_scores/Dolly/SafeLoRA-dolly/dolly-Ministral-8b-lora-safelora/"

# AsFT
# python gen_model_answer.py --model-id "Ministral-3-dolly-AsFT-reg1" --model-path "/workspace/models/Ministral-3-8B-Instruct-textonly" --use-lora --adapter-path "../../../experiments/AsFT/finetuned_models/dolly/Ministral-3-8B-Instruct-textonly/dolly_AsFT_reg1_top100_clean/"

# SafeGrad
# python gen_model_answer.py --model-id "Ministral-3-dolly-SafeGrad-rho1" --model-path "/workspace/models/Ministral-3-8B-Instruct-textonly" --use-lora --adapter-path "../../../experiments/SafeGrad/ckpt/dolly/Ministral-3-8B-Instruct-textonly_safegrad_f_1_0_100_100/"

# Lisa
# python gen_model_answer.py --model-id "llama-3-dolly-lisa-alignment-10-finetune-10" --model-path "/workspace/models/Llama-3.1-8B" --use-lora --adapter-path "../../../experiments/Lisa/ckpt/dolly/Llama-3.1-8B_lisa_f_1_0_100_10_10_10000/"

# Ours SMART
# python gen_model_answer.py --model-id "llama-3-dolly-SMART-mu2" --model-path "/workspace/models/Llama-3.1-8B-Instruct" --use-lora --adapter-path "../../../experiments/bppo_finetuning/Dolly/dolly_lora_ft_smart_kl_mu2/20/"
#python gen_model_answer.py --model-id "gemma-3-dolly-SMART-mu2" --model-path "/workspace/models/gemma-3-4b-it-textonly" --use-lora --adapter-path "../../../experiments/bppo_finetuning/Dolly/gemma-3-4b-it-textonly/dolly_lora_ft_smart_kl_mu2/20/"

# # Alpaca dataset
# python gen_model_answer_lisa.py --model-id "llama-2-alpaca-SafeGrad-rho4" --model-path "/workspace/models/Llama-2-7B-Chat-fp16" --use-lora --adapter-path "../../../experiments/SafeGrad/ckpt/alpaca/Llama-2-7B-Chat-fp16_safegrad_f_4_0_100_100/"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list Ministral-3-dolly-SafeInstr-k10