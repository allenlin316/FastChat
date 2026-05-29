# Generate MT-bench answers + GPT judgment for Full-FT Llama-2-7B models (seed 42)
#   datasets: dolly / alpaca only
#   methods : Benign-data-breaks-safety, Ours-Full-final_min5_cand10
# Run from FastChat/fastchat/llm_judge/ (paths are relative: ../../../experiments/...)
# NOTE: Full-FT models -> model path ends in /${seed} (NOT /${seed}_merged).
for seed in 42; do

model_ids=""

for dataset in dolly alpaca; do
    case $dataset in
        dolly)  dataset_cap="Dolly";  model_subdir="dolly-7b-full-ft"  ;;
        alpaca) dataset_cap="Alpaca"; model_subdir="alpaca-7b-full-ft" ;;
    esac

    ## SFT (w/o defense): Benign-data-breaks-safety (Full FT)
    benign_id="llama-2-benign-data-breaks-safety-${dataset}-seed${seed}"
    python gen_model_answer.py --model-id ${benign_id} \
        --model-path "../../../experiments/1.1_harmful_scores/${dataset_cap}/Benign-data-breaks-safety/${model_subdir}/${seed}/"

    ## Ours (Full FT, min5_cand10)
    ours_id="llama-2-ours-full-final-min5-cand10-${dataset}-seed${seed}"
    python gen_model_answer.py --model-id ${ours_id} \
        --model-path "../../../experiments/1.1_harmful_scores/${dataset_cap}/Ours-Full-final_min5_cand10/${model_subdir}/${seed}/"

    model_ids="${model_ids} ${benign_id} ${ours_id}"
done

## gen judgment on gpt-5.1 (all generated models at once)
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list ${model_ids}

done
