# Generate MT-bench answers + GPT judgment for Full-FT Llama-2-7B Full-benign models (seed 106)
#   datasets: dolly / alpaca only
#   method  : Full-benign
# Run from FastChat/fastchat/llm_judge/ (paths are relative: ../../../experiments/...)
# NOTE: Full-FT models -> model path ends in /${seed} (NOT /${seed}_merged).
for seed in 106; do

model_ids=""

for dataset in dolly alpaca; do
    case $dataset in
        dolly)  dataset_cap="Dolly";  model_subdir="dolly-7b-full-ft"  ;;
        alpaca) dataset_cap="Alpaca"; model_subdir="alpaca-7b-full-ft" ;;
    esac

    ## Full-benign (Full FT)
    full_benign_id="llama-2-full-benign-${dataset}-seed${seed}"
    python gen_model_answer.py --model-id ${full_benign_id} \
        --model-path "../../../experiments/1.1_harmful_scores/${dataset_cap}/Full-benign/${model_subdir}/${seed}/"

    model_ids="${model_ids} ${full_benign_id}"
done

## gen judgment on gpt-5.1 (all generated models at once)
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list ${model_ids}

done
