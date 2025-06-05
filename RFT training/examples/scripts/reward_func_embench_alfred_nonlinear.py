import os, re, json, ast
from datetime import datetime
from typing import Dict, Any, Tuple, List

import torch

# ─────────────────── 读取全局动作映射 ─────────────────── #
ACTION_MAP_PATH = '/root/autodl-tmp/datasets/alfred/alfred_action_map.json'
with open(ACTION_MAP_PATH, "r", encoding="utf-8") as fp:
    ACTION_MAP: Dict[str, str] = json.load(fp)   # {"0": "goto alarmclock", ...}

ACTION_IDS = set(ACTION_MAP.keys())

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

# ─────────────────── 帮助函数 ─────────────────── #
def parse_json_output(text: str) -> Tuple[dict, str]:
    try:
        return json.loads(text.strip()), ""
    except Exception as e:
        return {}, f"json error: {e}"

def get_response_from_query(q: str) -> str:
    m = re.search(r"<\|im_start\|>assistant\n", q)
    if not m:
        return ""
    resp = q[m.end():]
    for end_token in ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]:
        resp = resp.replace(end_token, "")
    return resp.strip()

def get_query_from_query(q: str) -> str:
    pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    try:
        return re.findall(pattern, q, re.DOTALL)[0]
    except Exception:
        return q

# ─────────────────── 格式奖励 ─────────────────── #
def check_structure(data: Dict[str, Any]) -> float:
    required = {"reasoning_and_reflection", "visual_state_description",
                "language_plan", "executable_plan"}
    return 0.125 if required.issubset(data) else 0.0

def check_executable_items(plan: List[Dict[str, Any]]) -> float:
    """
    0.25 × (合法条目 / 总条目)
    条目合法条件：有 int action_id + str action_name
    """
    if not plan:
        return 0.0
    good = sum(
        isinstance(step, dict)
        and isinstance(step.get("action_id"), int)
        and isinstance(step.get("action_name"), str)
        for step in plan
    )
    return 0.125 * good / len(plan)

def check_action_id_match(plan: List[Dict[str, Any]]) -> float:
    """
    0.25 × (id/name 对应正确条目数 / 总条目)
    """
    if not plan:
        return 0.0
    good = 0
    for step in plan:
        aid = step.get("action_id")
        name = step.get("action_name", "")
        if isinstance(aid, int) and (s := str(aid)) in ACTION_IDS:
            if name.lower() == ACTION_MAP[s].lower():
                good += 1
    return 0.25 * good / len(plan)

def format_reward(response: str) -> float:
    data, err = parse_json_output(response)
    if err:
        return 0.0
    plan = data.get("executable_plan", [])
    return (check_structure(data)
            + check_executable_items(plan)
            + check_action_id_match(plan))

# ─────────────────── 准确率奖励（多步） ─────────────────── #
def accuracy_reward(response: str,
                    answer: str) -> Tuple[float, List[str]]:
    """
    answer 示例: ['Goto handtowelholder', 'Pickup handtowel', ...]
    直接比较 action_name 字符串（忽略大小写）
    """
    data, err = parse_json_output(response)
    if err:
        return 0.0, []

    pred_plan = data.get("executable_plan", [])
    pred_names = [step.get("action_name", "").lower()
                  for step in pred_plan if isinstance(step, dict)]

    gold_raw = ast.literal_eval(answer)
    gold = [g.lower() for g in gold_raw]

    # 计算前缀正确率 (prefix accuracy)
    correct = 0
    for p, g in zip(pred_names, gold):
        if p == g:
            correct += 1
        else:
            break

    if not gold:
        return 0.0, pred_names

    total = len(gold)
    reward = (correct * (correct + 1) / 2) / (total * (total + 1) / 2)
    if total == 1:
        if len(pred_plan) > 1:
            reward = reward - 0.25
    return reward, pred_names

# ─────────────────── 主调 ─────────────────── #
def reward_func(queries, prompts, labels):
    ts = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards, acc_r, fmt_r = [], [], []

    with open(LOG_PATH, "a") as log:
        log.write(f"------------------------- {ts} -------------------------\n")
        for q, p, label in zip(queries, prompts, labels):
            try:
                resp = get_response_from_query(q)
                if not resp:
                    rewards.append(0.0)
                    acc_r.append(0.0)
                    fmt_r.append(0.0)
                    log.write("empty response\n")
                    continue

                acc, parsed = accuracy_reward(resp, label)
                fmt = format_reward(resp)

                rewards.append(acc + fmt)
                acc_r.append(acc)
                fmt_r.append(fmt)

                q_clean = get_query_from_query(q)
                log.write("=======================================================\n")
                log.write("Query: " + q_clean + "\n")
                log.write("Response: " + resp + "\n")
                log.write("Label: " + str(label) + "\n")
                log.write(f"Parsed Prediction: {parsed}\n")
                log.write(f"Accuracy: {acc:.2f} | Format: {fmt:.2f}\n")
                log.write("=======================================================\n\n")
            except Exception as e:
                log.write(f"Exception: {e}\n")
                rewards.append(0.0)
                acc_r.append(0.0)
                fmt_r.append(0.0)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(acc_r, dtype=torch.float32),
        "format_rewards": torch.tensor(fmt_r, dtype=torch.float32),
    }
