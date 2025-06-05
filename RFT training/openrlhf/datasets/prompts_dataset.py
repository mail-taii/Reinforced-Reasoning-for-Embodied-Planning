from typing import Any, Dict

from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data_item, input_template=None, input_key="input", apply_chat_template=None) -> Dict[str, Any]:
    assert apply_chat_template
    assert "conversations" in data_item
    assert "image_urls" in data_item

    conversations = data_item["conversations"]
    assert len(conversations) == 2
    prompt = apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    data = {
        "prompt": prompt,
        **data_item,
    }
    return data


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.data = []
        for data_item in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            d = preprocess_data(data_item, input_template, input_key, apply_chat_template)
            self.data.append(d)

    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, idx):
        return self.data[idx]
