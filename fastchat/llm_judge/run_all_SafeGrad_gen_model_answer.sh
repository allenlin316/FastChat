# Generate all methods

## Dolly dataset
python gen_model_answer.py --model-id "llama-2-dolly-SafeGrad-rho1-seed106" --model-path "../../../experiments/SafeGrad/ckpt/dolly/Llama-2-7B-Chat-fp16_safegrad_f_1_0_100_100_seed106/106_merged/"

## Alpaca dataset
python gen_model_answer.py --model-id "llama-2-alpaca-SafeGrad-rho1-seed106" --model-path "../../../experiments/SafeGrad/ckpt/alpaca/Llama-2-7B-Chat-fp16_safegrad_f_1_0_100_100_seed106/106_merged/"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-dolly-SafeGrad-rho1-seed106 llama-2-alpaca-SafeGrad-rho1-seed106
