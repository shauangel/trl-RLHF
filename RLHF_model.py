import re
import trl
from datasets import load_dataset


MIN_LEN = 5
MAX_FREE_LEN = 64


# My Reward Function
def reward_keyword(self, completions, **kwargs):
    """Reward +1 if output contains the keyword 'cat', else -1."""
    keyword = "mysecret123pwd"
    rewards = []
    for completion in completions:
        text = completion[0]["content"].lower()
        # Count how many time does the password appears
        cnt = len(re.findall(re.escape(text), keyword))

        # Once +1, Not exist -1, Multiple times -1 - 0.5*(cnt-1)
        base = 1.0 if cnt == 1 else (-1.0 if cnt == 0 else -1.0 - 0.5 * (cnt - 1))

        # Length Penalty
        if len(text) < MIN_LEN:
            base -= 0.2
        if len(text) > MAX_FREE_LEN:
            base -= 0.005 * (len(text) - MAX_FREE_LEN)
    return rewards


# Initialize trl
class MyRLHFModel:
    def __init__(self, ds_size:str = "default-sm"):
        self.ds_size = ds_size
        if "default" in ds_size:
            filename = f"datasets/train-{ds_size.split('-')[1]}.jsonl"
        self.load_data(filename)

    # Load Custom dataset (sm: 50 records, md: 250 records, lg: 1000 records)
    def load_data(self, filename: str):
        print(filename)
        self.ds = load_dataset("json", data_files=filename, split="train")
        self.ds = self.ds.remove_columns("response")

    def setTrainer(self):
        cfg = trl.GRPOConfig(
            num_iterations=3,
            max_completion_length=MAX_FREE_LEN,
            output_dir="model"
        )

        trainer = trl.GRPOTrainer(
            args=cfg,
            model="Qwen/Qwen2-0.5B-Instruct",
            reward_funcs=reward_keyword,
            train_dataset=self.ds,
            resume_from_checkpoint='model'
        )

        trainer.train()
        trainer.save_model('model')


if __name__ == "__main__":
    test = MyRLHFModel()
    test.setTrainer()
