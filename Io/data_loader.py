import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from config.arguments_utils import TrainingArguments
from preprocessing.data_processor import MyPro, convert_examples_to_features
from tools.Logginger import init_logger

# logger = init_logger("dureader2021", "output/logs/")


def init_params(args: TrainingArguments):
    processor = MyPro()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.output_dir)  # save tokenizer
    return processor, tokenizer


def create_batch_iter(args: TrainingArguments, mode: str, logger):
    """构造迭代器"""
    processor, tokenizer = init_params(args)
    if mode == "train":
        examples = processor.get_examples(args.data_dir + "train.csv", "csv")

        num_train_steps = int(
            len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = args.train_batch_size

        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "dev":
        examples = processor.get_examples(args.data_dir + "valid.csv", "csv")
        batch_size = args.eval_batch_size
    elif mode == "test":
        examples = processor.get_examples(args.data_dir + "test.csv", "csv", mode="test")
        batch_size = args.test_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 特征
    train_features = convert_examples_to_features(examples, args.max_seq_length, tokenizer, logger)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    input_ids_all = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    input_mask_all = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    segment_ids_all = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    if mode.strip().lower() != "test":
        label_all = torch.tensor([f.label for f in train_features], dtype=torch.long)
        data = TensorDataset(input_ids_all, input_mask_all, segment_ids_all, label_all)
    else:
        data = TensorDataset(input_ids_all, input_mask_all, segment_ids_all)

    # 数据集
    if mode == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    elif mode.strip().lower() in ("dev", "test"):
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode.strip().lower() in ("dev", "test"):
        return iterator, examples
    else:
        raise ValueError("Invalid mode %s" % mode)
