"""Generate answers with local models.
Modified to support Lisa two-LoRA setup where training resizes vocab to 32001 (pad token added).

Usage:
python3 gen_model_answer_lisa.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0

LoRA Usage:
python3 gen_model_answer_lisa.py \
    --model-path <base_model_path> \
    --model-id <model_id> \
    --use-lora \
    --adapter-path <path_to_lora_adapter>

QLoRA Usage:
python3 gen_model_answer_lisa.py \
    --model-path <base_model_path> \
    --model-id <model_id> \
    --use-qlora \
    --adapter-path <path_to_qlora_adapter>
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.conversation import get_conv_template
from fastchat.utils import str_to_torch_dtype


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    use_lora=False,
    use_qlora=False,
    adapter_path=None,
    conv_template=None,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                use_lora=use_lora,
                use_qlora=use_qlora,
                adapter_path=adapter_path,
                conv_template=conv_template,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    use_lora=False,
    use_qlora=False,
    adapter_path=None,
    conv_template=None,
):
    if use_lora:
        # Load model with LoRA (without quantization)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading LoRA model from {model_path}")

        # Load base model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype if dtype else torch.bfloat16,
            trust_remote_code=True,
        )

        # Lisa training adds a pad token, resizing vocab from 32000 to 32001.
        # Resize here to match before loading any LoRA.
        model.resize_token_embeddings(32001, mean_resizing=False)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load LoRA adapter if specified
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                torch_dtype=dtype if dtype else torch.bfloat16,
            )
            print("LoRA adapter loaded successfully")
        else:
            print("WARNING: No adapter path specified. Using base model only.")

        print("LoRA model loaded successfully")

        # Enable KV cache for efficient generation (disable causes O(n²) memory growth)
        model.config.use_cache = True
    elif use_qlora:
        # Load model with QLoRA
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading QLoRA model from {model_path} with adapter {adapter_path}")

        # Configure 4-bit quantization for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype if dtype else torch.float16,
            trust_remote_code=True,
        )

        # Lisa training adds a pad token, resizing vocab from 32000 to 32001.
        # Resize here to match before loading any LoRA.
        model.resize_token_embeddings(32001, mean_resizing=False)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load LoRA adapter if specified
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                torch_dtype=dtype if dtype else torch.bfloat16,
            )
            print("LoRA adapter loaded successfully")

        print("QLoRA model loaded successfully")

        # Enable KV cache for efficient generation
        model.config.use_cache = True
    else:
        # Load full precision model (same way as LoRA for consistency)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_dtype = dtype if dtype else torch.bfloat16
        print(f"Loading full precision model from {model_path}")
        print(f"Using dtype: {load_dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=load_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Lisa training adds a pad token, resizing vocab from 32000 to 32001.
        # Resize here to match before loading any LoRA.
        model.resize_token_embeddings(32001, mean_resizing=False)

        # Print actual model dtype and memory usage
        print(f"Model dtype: {model.dtype}")
        print(f"Model device: {model.device}")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

        # Enable KV cache for efficient generation (disable causes O(n²) memory growth)
        model.config.use_cache = True

        print("Full precision model loaded successfully")

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            # Use specified conv_template or fall back to model_id based template
            if conv_template:
                conv = get_conv_template(conv_template)
            else:
                conv = get_conversation_template(model_id)
            turns = []
            # Only evaluate Turn 1 (single-turn), skip multi-turn to match gen_judgment.py
            for j in range(min(1, len(question["turns"]))):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    # Move input to model's device (handles device_map="auto")
                    input_tensor = torch.as_tensor(input_ids).to(model.device)
                    output_ids = model.generate(
                        input_tensor,
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    raw_output = tokenizer.decode(output_ids, skip_special_tokens=False)

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA (full precision) for loading the model.",
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization) for loading the model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to the LoRA adapter weights (required when using --use-lora or --use-qlora).",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        help="Conversation template to use (e.g., 'zero_shot', 'vicuna_v1.1', 'llama-2', 'raw'). If not set, uses model_id to determine template.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # Validate LoRA/QLoRA arguments
    if args.use_lora and args.use_qlora:
        print("ERROR: Cannot use both --use-lora and --use-qlora. Please choose one.")
        exit(1)

    if args.use_lora:
        print("=" * 60)
        print("LoRA Mode Enabled (Full Precision) [Lisa: vocab=32001]")
        print("=" * 60)
        print(f"Base model: {args.model_path}")
        if args.adapter_path:
            print(f"LoRA adapter: {args.adapter_path}")
        else:
            print("WARNING: No adapter path specified. Loading base model only.")
        print("=" * 60)

    if args.use_qlora:
        print("=" * 60)
        print("QLoRA Mode Enabled (4-bit Quantization) [Lisa: vocab=32001]")
        print("=" * 60)
        print(f"Base model: {args.model_path}")
        if args.adapter_path:
            print(f"LoRA adapter: {args.adapter_path}")
        else:
            print("WARNING: No adapter path specified. Loading base model only.")
        print("=" * 60)

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        adapter_path=args.adapter_path,
        conv_template=args.conv_template,
    )

    reorg_answer_file(answer_file)
