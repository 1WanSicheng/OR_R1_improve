# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config
import os
import tempfile
import subprocess
import torch
from datetime import datetime

from structure_reward import score_structure


@dataclass
class DataConfig:
    dataset_path: str = ""


@dataclass
class StructRewardConfig:
    lambda_struct: float = 1.0
    alpha_obj: float = 1.0
    alpha_var: float = 1.0
    alpha_con: float = 2.0
    alpha_align: float = 1.0
    log_dir: str = "./logs_struct"


TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""


def compile_script(script_content, timeout=10):
    target_dir = "./eval_execute"
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir=target_dir) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content.encode())
    try:
        process = subprocess.run(
            ["python", tmp_file_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True,
        )
        execution_result = process.stdout
        execution_best_solution_start_pos = execution_result.find("Just print the best solution:")
        if execution_best_solution_start_pos != -1:
            execution_best_solution = execution_result[execution_best_solution_start_pos:].replace(
                "Just print the best solution:", ""
            ).strip()
            execution_best_solution_end_pos = execution_best_solution.find("\n")
            if execution_best_solution_end_pos != -1:
                execution_best_solution = execution_best_solution[:execution_best_solution_end_pos]
            execution_state = "Execution Successful and Best Solution Found"
        else:
            if "No Best Solution" in execution_result:
                execution_best_solution = "No Best Solution"
                execution_state = "Execution Successful but No Best Solution Found"
            else:
                execution_best_solution = None
                execution_state = "Execution Suceessful but Out of Expectation"
    except subprocess.TimeoutExpired as e:
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = "Execution Failed: Timeout"
    except subprocess.CalledProcessError as e:
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = f"Execution Failed: {e.stdout}"
    finally:
        os.remove(tmp_file_name)

    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution,
        "execution_state": execution_state,
    }


def run_code(output):
    add_script = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'
    start = output.find("```python")
    if start == -1:
        return None
    end = output.find("```", start + 9)
    script = output[start:end].replace("```python", "") + add_script
    return compile_script(script)


parser = HfArgumentParser((GRPOConfig, ModelConfig, DataConfig, StructRewardConfig))
grpo_args, model_args, data_args, struct_args = parser.parse_args_into_dataclasses()


def reward_with_reference(completions, **kwargs):
    format_rewards = []
    valid_code_rewards = []
    answer_rewards = []
    structure_rewards = []

    true_count = 0
    records = []
    gpu_id = torch.cuda.current_device()

    prediction_answers = []
    for completion in completions:
        format_reward = 0.0
        formats = [
            "## Mathematical Model:",
            "## Decision Variables:",
            "## Objective Function:",
            "## Constraints:",
            "## Python Code Solution Using `coptpy`:",
            "```python",
        ]
        for marker in formats:
            if completion.find(marker) != -1:
                format_reward += 1
        format_rewards.append(format_reward / len(formats))

    for completion in completions:
        valid_code_reward = 0.0
        prediction_execution_output = run_code(completion)
        if prediction_execution_output is None:
            prediction_answers.append(None)
        else:
            prediction_answers.append(prediction_execution_output["execution_best_solution"])
            if prediction_execution_output["execution_best_solution"] is not None:
                valid_code_reward = 1
        valid_code_rewards.append(valid_code_reward)

    voting_group_size = getattr(grpo_args, "num_generations", 8)
    voting_answers = []
    for i in range(0, len(completions), voting_group_size):
        prediction_answer_dict = {}
        for j in range(voting_group_size):
            if i + j >= len(prediction_answers):
                break
            answer = prediction_answers[i + j]
            if answer is None or answer == "No Best Solution":
                continue
            try:
                key = int(float(answer))
            except ValueError:
                continue
            prediction_answer_dict[key] = prediction_answer_dict.get(key, 0) + 1

        voting_answer = None
        max_count = 1
        for key, count in prediction_answer_dict.items():
            if count > max_count:
                max_count = count
                voting_answer = key

        for _ in range(voting_group_size):
            voting_answers.append(voting_answer)
    voting_answers = voting_answers[: len(completions)]

    for i, completion in enumerate(completions):
        answer_reward = 0.0
        gt_answer = kwargs["answer"][i]
        question = kwargs["question"][i]
        voting_answer = voting_answers[i] if i < len(voting_answers) else None
        prediction_answer = prediction_answers[i]

        try:
            if prediction_answer is None or prediction_answer == "No Best Solution":
                answer_reward = 0.0
            elif voting_answer is None:
                answer_reward = 0.0
            elif int(float(prediction_answer)) == voting_answer:
                answer_reward = 1.0
                true_count += 1
        except ValueError:
            answer_reward = 0.0

        answer_rewards.append(answer_reward)

        structure_breakdown = score_structure(
            question,
            completion,
            alpha_obj=struct_args.alpha_obj,
            alpha_var=struct_args.alpha_var,
            alpha_con=struct_args.alpha_con,
            alpha_align=struct_args.alpha_align,
        )
        structure_rewards.append(structure_breakdown["r_struct"])

        records.append(
            [
                gpu_id,
                i,
                prediction_answer,
                voting_answer,
                gt_answer,
                format_rewards[i],
                valid_code_rewards[i],
                answer_rewards[i],
                structure_breakdown["r_obj"],
                structure_breakdown["r_var"],
                structure_breakdown["r_con"],
                structure_breakdown["r_align"],
                structure_breakdown["r_struct"],
            ]
        )

    print(
        "True Rate: "
        f"{true_count/len(completions):.2f}, "
        f"Mean Format Reward: {sum(format_rewards)/len(format_rewards):.2f}, "
        f"Mean Valid Code Reward: {sum(valid_code_rewards)/len(valid_code_rewards):.2f}, "
        f"Mean Answer Reward: {sum(answer_rewards)/len(answer_rewards):.2f}, "
        f"Mean Struct Reward: {sum(structure_rewards)/len(structure_rewards):.2f}"
    )

    os.makedirs(struct_args.log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(struct_args.log_dir, "records.csv"), "a") as f:
        f.writelines("\n")
        for record in records:
            f.writelines(current_time + "," + ",".join(str(item) for item in record) + "\n")

    torch.cuda.empty_cache()
    rewards = []
    for i in range(len(completions)):
        reward = 0.0
        reward += format_rewards[i]
        reward += valid_code_rewards[i]
        reward += answer_rewards[i]
        reward += struct_args.lambda_struct * structure_rewards[i]
        rewards.append(reward)
    return rewards


model_path = model_args.model_name_or_path
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def format_dataset(example):
    prompt = TEMPLATE_q2mc_en.replace("{Question}", example["question"].strip()).strip()
    example["prompt"] = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    return example


dataset = load_dataset("json", data_files=data_args.dataset_path)
formatted_dataset = dataset.map(format_dataset)

grpo_trainer = GRPOTrainer(
    model,
    args=grpo_args,
    reward_funcs=reward_with_reference,
    train_dataset=formatted_dataset["train"],
    processing_class=tokenizer,
    peft_config=get_peft_config(model_args),
)

grpo_trainer.train()
grpo_trainer.save_model(grpo_args.output_dir)
