# Generate all methods

## Dolly dataset
python gen_model_answer.py --model-id "llama-2-dolly-SAP-seed20" --model-path "../../../experiments/SAP/SAPcode/ckpts/Llama-2-7B-Chat-fp16/sap_model_dolly_seed20/20_merged/"

## Alpaca dataset
python gen_model_answer.py --model-id "llama-2-alpaca-SAP-seed20" --model-path "../../../experiments/SAP/SAPcode/ckpts/Llama-2-7B-Chat-fp16/sap_model_alpaca_seed20/20_merged/"

## gen judgment on gpt-5.1
echo "" | python gen_judgment.py --judge-model gpt-5.1 --model-list llama-2-dolly-SAP-seed20 llama-2-alpaca-SAP-seed20 llama-2-gsm8k-SAP-seed20