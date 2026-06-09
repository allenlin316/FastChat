# Generate model answers for the 8 Dolly SMART runs (λ × seed), then judge with GPT-5.1.

EXP_LOCAL="../../../experiments"
SUBDIR="bppo_finetuning/Dolly/Llama-2-7B-Chat-fp16"

MODEL_IDS=()

for seed in 106; do
for lambda in 0 0.3 0.5 1.0; do
    method="dolly_lora_ft_smart_kl_lambda${lambda}_mu10_lr2e-5_ep10"
    model_id="llama-2-dolly-lambda${lambda}-mu10-seed${seed}"
    local_path="${EXP_LOCAL}/${SUBDIR}/${method}/${seed}_merged"

    python gen_model_answer.py --model-id "$model_id" --model-path "$local_path"
    MODEL_IDS+=("$model_id")
done
done

#echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list "${MODEL_IDS[@]}"
