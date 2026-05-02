# Generate all methods

## Alpaca dataset
align_steps=(18 14 10 6 2)
finetune_steps=(2 6  10 14 18)
for i in "${!align_steps[@]}"; do
    align_step=${align_steps[$i]}
    finetune_step=${finetune_steps[$i]}
    echo "=========================================="
    echo "Running: align_step=${align_step}, finetune_step=${finetune_step}"
    echo "=========================================="

    python gen_model_answer_lisa.py --model-id "llama-2-alpaca-lisa-alignment-${align_step}-finetune-${finetune_step}" --model-path "/workspace/models/Llama-2-7b-hf" --use-lora --adapter-path "../../../experiments/Lisa/ckpt/alpaca/Llama-2-7b-hf_lisa_f_1_0_100_${align_step}_${finetune_step}_10000/"

    echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-alpaca-lisa-alignment-${align_step}-finetune-${finetune_step}

done