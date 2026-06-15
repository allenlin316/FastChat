# NOTE: SafeInstr seed 106 checkpoint will be added later.

for seed in 20 42 106; do
    echo "Run SFT method MT-Bench (seed=${seed})"
    python gen_model_answer.py --model-id "llama-3.1-alpaca-SFT-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Alpaca/Ours-lora-final_min5_cand10/alpaca-8b-lora-ft/${seed}_merged/"

    echo "Run SafeInstr MT-Bench (seed=${seed})"
    python gen_model_answer.py --model-id "llama-3.1-alpaca-SafeInstr-Bianchi-k10-seed${seed}" --model-path "../../../experiments/1.1_harmful_scores/Alpaca/Bianchi-safety-augmentation_k10/alpaca-8b-lora-ft/${seed}_merged/"
done

echo "Starting running LLM-as-judge"

for seed in 20 42 106; do
    echo "Judging seed=${seed} (SFT, SafeInstr)"
    echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-3.1-alpaca-SFT-seed${seed} llama-3.1-alpaca-SafeInstr-Bianchi-k10-seed${seed}
done
