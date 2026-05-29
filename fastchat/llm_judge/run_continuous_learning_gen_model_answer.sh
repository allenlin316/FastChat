# Generate all methods
for seed in 20; do

## without 1st stage (LoRA, lr=2e-5)
# python gen_model_answer.py --model-id llama-2-continuous-learning-without-first-stage-dolly-seed${seed} --model-path "../../../experiments/1.1_harmful_scores/Dolly/Dolly-2000/dolly-7b-lora-ft/${seed}_merged/"

## Full FT + LoRA FT (lr=2e-5)
# python gen_model_answer.py --model-id llama-2-continuous-learning-two-stage-dolly-dolly-seed${seed} --model-path "../../../experiments/1.3_data_poisoning/Dolly-Dolly/dolly-7b-full-ft-dolly-lora-2000-lr2e-5-epoch1/${seed}-second_merged/"

## Ours-LoRA (lr=2e-5)
python gen_model_answer.py --model-id llama-2-continuous-learning-ours-defense-dolly-dolly-mu6-lr2e-5-seed${seed} --model-path "../../../experiments/bppo_finetuning/Dolly-Dolly/Llama-2-7B-Chat-fp16/dolly-dolly_lora_ft_smart_kl_lambda0_mu6_lr2e-5_anchor-Llama-2-7B-Chat-fp16_fix1_bianchi_k100/20_merged/"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-continuous-learning-ours-defense-dolly-dolly-mu6-lr2e-5-seed${seed}

done
