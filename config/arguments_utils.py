from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    Training arguments
    """
    model: Optional[str] = field(
        # default="/mnt/model/pt/chinese_roformer_base",
        default="C:\dh\model\pt\chinese_roformer_base",
        metadata={"help": "The pre-trained model dir."})
    data_dir: Optional[str] = field(
        default="C:\dh\souhu_text_match_2021\data\clean\demo",
        metadata={"help": "The input data dir."})
    output_dir: str = field(
        default="output/round1/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    num_train_epochs: int = field(
        default=1,  # 4
        metadata={"help": "Total number of training epochs to perform."})
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after WordPiece tokenization. \n"
                          "Sequences longer than this will be truncated, and sequences shorter \n"
                          "than this will be padded."})
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the dev set."})
    train_batch_size: int = field(
        default=8,
        metadata={"help": "Total batch size for training."})
    eval_batch_size: int = field(
        default=8,
        metadata={"help": "Total batch size for eval."})
    test_batch_size: int = field(
        default=256,
        metadata={"help": "Total batch size for eval."})
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": 'weight decay'})

    warmup_proportion: float = field(
        default=0.1,
        metadata={"help": "Proportion of training to perform linear learning rate warmup for. "
                          "E.g., 0.1 = 10%% of training."})
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Whether not to use CUDA when available"})
    local_rank: int = field(
        default=-1,
        metadata={"help": "Distributed "})
    seed: int = field(
        default=42,
        metadata={"help": "random seed for initialization"})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    eval_step: int = field(default=1)  # 2000
    do_adversarial: bool = field(
        default=False,
        metadata={"help": "Whether join adversarial in training"})
