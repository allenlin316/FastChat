for seed in 20; do
    echo "Run SAP method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SAP-seed${seed}" --model-path "../../../experiments/SAP/SAPcode/ckpts/Llama-3.1-8B-Instruct/sap_model_dolly_seed20/20_merged/"
    
    echo "Run SafeLoRA method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SafeLoRA-threshold0.8-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/SafeLoRA-dolly/dolly-8b-lora-safelora/20_merged/"
    
    echo "Run LISA method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-LISA-seed${seed}" --model-path "../../../experiments/Lisa/ckpt/dolly/Llama-3.1-8B_lisa_f_1_0_100_10_10_10000/20_merged/"
    
    echo "Run AsFT method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-AsFT-seed${seed}" --model-path "../../../experiments/AsFT/finetuned_models/dolly/Llama-3.1-8B-Instruct/dolly_AsFT_reg1_top100_clean/20_merged/"
    
    echo "Run SafeGrad method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SafeGrad-seed${seed}" --model-path "../../../experiments/SafeGrad/ckpt/dolly/Llama-3.1-8B-Instruct_safegrad_f_1_0_100_100_seed20/20_merged/"
    
    echo "Run Ours method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly_lora_ft_smart_kl_lambda0.5_mu10-seed${seed}" --model-path "../../../experiments/bppo_finetuning/Dolly/Llama-3.1-8B-Instruct/dolly_lora_ft_smart_kl_lambda0.5_mu10/20_merged/"
done

# ---------------------------------------------------------------------------------------------------------------------------------------------------

for seed in 42; do

echo "Run SFT method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SFT-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-lora-final_min5_cand10/dolly-8b-lora-ft/42_merged/"

    echo "Run SafeInstr MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SafeInstr-Bianchi-k10-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Bianchi-safety-augmentation_k10/dolly-8b-lora-ft/42_merged/"

    echo "Run Ours method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly_lora_ft_smart_kl_lambda0.5_mu10-seed${seed}" --model-path "../../../experiments/bppo_finetuning/Dolly/Llama-3.1-8B-Instruct/dolly_lora_ft_smart_kl_lambda0.5_mu10/42_merged/"
done

for seed in 106; do
    echo "Run SFT method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SFT-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Ours-lora-final_min5_cand10/dolly-8b-lora-ft/106_merged/"

    echo "Run SafeInstr MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SafeInstr-Bianchi-k10-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/Bianchi-safety-augmentation_k10/dolly-8b-lora-ft/106_merged/"
    
    echo "Run SAP method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SAP-seed${seed}" --model-path "../../../experiments/SAP/SAPcode/ckpts/Llama-3.1-8B-Instruct/sap_model_dolly_seed106/106_merged/"
    
    echo "Run SafeLoRA method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SafeLoRA-threshold0.8-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Dolly/SafeLoRA-dolly/dolly-8b-lora-safelora/106_merged/"
    
    echo "Run LISA method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-LISA-seed${seed}" --model-path "../../../experiments/Lisa/ckpt/dolly/Llama-3.1-8B_lisa_f_1_0_100_10_10_10000/106_merged/"
    
    echo "Run AsFT method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-AsFT-seed${seed}" --model-path "../../../experiments/AsFT/finetuned_models/dolly/Llama-3.1-8B-Instruct/dolly_AsFT_reg1_top100_clean/106_merged/"
    
    echo "Run SafeGrad method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly-SafeGrad-seed${seed}" --model-path "../../../experiments/SafeGrad/ckpt/dolly/Llama-3.1-8B-Instruct_safegrad_f_1_0_100_100_seed106/106_merged/"
    
    echo "Run Ours method MT-Bench"
    python gen_model_answer.py --model-id "llama-3.1-dolly_lora_ft_smart_kl_lambda0.5_mu10-seed${seed}" --model-path "../../../experiments/bppo_finetuning/Dolly/Llama-3.1-8B-Instruct/dolly_lora_ft_smart_kl_lambda0.5_mu10/106_merged/"
done

echo "Starting running LLM-as-judge"
echo "Starting with seed=42 only with SFT, SafeInstr and Ours method"
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-3.1-dolly-SFT-seed${seed} llama-3.1-dolly-SafeInstr-Bianchi-k10-seed${seed} llama-3.1-dolly_lora_ft_smart_kl_lambda0.5_mu10-seed${seed}


echo "Start with seed=106 only method (SFT, SafeInstr)"
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-3.1-dolly-SFT-seed${seed} llama-3.1-dolly-SafeInstr-Bianchi-k10-seed${seed}

echo "Run remaining methods"
for seed in 20 106; do
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-3.1-dolly-SAP-seed${seed} llama-3.1-dolly-SafeLoRA-threshold0.8-seed${seed} llama-3.1-dolly-LISA-seed${seed} llama-3.1-dolly-AsFT-seed${seed} llama-3.1-dolly-SafeGrad-seed${seed} llama-3.1-dolly_lora_ft_smart_kl_lambda0.5_mu10-seed${seed}
done