import os
import glob
import re
import json
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen3-14B"
GENE_DESC_FILE = 'genes_express.txt'
INPUT_DIR = "data"
OUTPUT_DIR_RESULT = "result"
OUTPUT_DIR_THINKING = "thinking_result"



def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def extract_genes_from_prompt(prompt):
    gene_pattern = r"'(\w+)'\s*:\s*\d+\.?\d*"
    genes = re.findall(gene_pattern, prompt)
    return list(set(genes))


def load_gene_descriptions(file_path):
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['gene']: item['description'] for item in data}
    except Exception:
        return {}


def build_gene_express_map(genes, description_dict):
    express_map = {}
    for gene in genes:
        description = description_dict.get(gene)
        if description and not description.endswith('\n'):
            express_map[gene] = f"{description}\n"
        else:
            express_map[gene] = description or "\n"
    return express_map


def parse_output(text):
    """
    Robust parsing: Handles cases where the <think> tag is unclosed.
    """
    if not text:
        return "", ""

    text = text.strip()

    # Check if there is a <think> tag
    if "<think>" in text:
        parts = text.split("<think>", 1)
        remainder = parts[1]

        if "</think>" in remainder:
            think_part, res_part = remainder.split("</think>", 1)
            return think_part.strip(), res_part.strip()
        else:
            return remainder.strip(), ""
    else:
        # No tags found, everything is considered as the main content
        return "", text


def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "data_*.txt")), key=natural_sort_key)
    if not input_files:
        print(f"No data_*.txt files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} files, preparing to process...")
    description_dict = load_gene_descriptions(GENE_DESC_FILE)
    prompts_data = []

    print("Building Prompts...")
    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_prompt = f.read()
            genes = extract_genes_from_prompt(raw_prompt)
            express = build_gene_express_map(genes, description_dict)

            # Strengthen the Prompt, enforcing the format
            prompt_content = f"""
            Question:
            {express}

            Please answer in a professional tone.
            IMPORTANT: Please think within the <think>...</think> tags first, and then output the final answer.
            Format as follows:
            <think>
            Write thinking process here...
            </think>
            Write the official answer here...
            
            """

            messages = [
                {"role": "system", "content": "You are a professional bioinformatics assistant."},
                {"role": "user", "content": prompt_content}
            ]

            prompts_data.append({"file": input_file, "messages": messages})
        except Exception as e:
            print(f"Preprocessing error: {e}")

    print(f"Loading model: {MODEL_ID} ...")
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=16000,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1,
        top_k=5,
        max_tokens=8192,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    print("Starting batch inference...")
    all_messages = [d["messages"] for d in prompts_data]
    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

    os.makedirs(OUTPUT_DIR_RESULT, exist_ok=True)
    os.makedirs(OUTPUT_DIR_THINKING, exist_ok=True)

    print("Saving results...")
    for i, output in enumerate(outputs):
        original_file = prompts_data[i]["file"]
        generated_text = output.outputs[0].text

        file_match = re.search(r'data_(\d+)\.txt', original_file)
        file_idx = file_match.group(1) if file_match else f"unknown_{i}"

        thinking, content = parse_output(generated_text)

        # Debug info: If the main text is empty, the generation might have been truncated
        if not content and thinking:
            print(
                f"Warning: data_{file_idx}.txt seems to only contain the thinking process and no main text (tag might be unclosed).")

        if thinking:
            with open(os.path.join(OUTPUT_DIR_THINKING, f"num_{file_idx}_thinking_result.txt"), "w",
                      encoding="utf-8") as f:
                f.write(thinking + "\n")

        with open(os.path.join(OUTPUT_DIR_RESULT, f"num_{file_idx}_result.txt"), "w", encoding="utf-8") as f:
            f.write(content + "\n")

        # print(f"[{i+1}/{len(outputs)}] Completed: data_{file_idx}.txt")

    print("All tasks processed successfully!")


if __name__ == "__main__":
    main()