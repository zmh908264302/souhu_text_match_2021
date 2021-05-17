import os

import torch
from roformer import RoFormerForSequenceClassification, RoFormerTokenizer
from tqdm import tqdm
from transformers import HfArgumentParser

from Io.data_loader import create_batch_iter
from config.arguments_utils import TrainingArguments
from tools.Logginger import init_logger
from tools.model_util import set_seed


def predict():
    parser = HfArgumentParser(TrainingArguments)
    args: TrainingArguments = parser.parse_args_into_dataclasses()[0]

    logger = init_logger("souhu-text-match-2021", "output/logs/")
    logger.info(f"!!!!!!Test arguments: {args}")

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1

    logger.info(f"device: {device}, n_gpu: {args.n_gpu}")

    set_seed(args)
    test_dataloader = create_batch_iter(args, "test", logger)

    args.output_dir = args.output_dir + sorted(os.listdir(args.output_dir))[-1]  # 最新一次训练结果
    logger.info(f"model {args.output_dir} predict useed")

    tokenizer = RoFormerTokenizer.from_pretrained("/home/zhuminghao/work/model/pt/chinese_roformer_base")  # 没保存，所以用原始一样
    model = RoFormerForSequenceClassification.from_pretrained(args.output_dir)
    model.to(device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        test_logits = []
        ids = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="test", ascii=True)):
            sources, targets, bt_ids = batch
            inputs = list(zip(sources, targets))
            ids.append(bt_ids)

            pt_batch = tokenizer(
                inputs,
                padding=True,
                truncation="longest_first",
                max_length=args.max_seq_length,
                return_tensors="pt"
            )
            pt_batch = pt_batch.to(device)

            outputs = model(**pt_batch, return_dict=True)

            logits = torch.max(outputs.logits, dim=1)[1]
            if device.type == "cuda":
                logits = logits.cpu().numpy().astype(int)
            else:
                logits = logits.numpy()
            test_logits.extend(logits.tolist())

        output_path = args.output_dir + "/test.csv"
        with open(output_path, "w", encoding="utf-8") as fw:
            for id, label in zip(ids, test_logits):
                fw.write(",".join([id, str(label)]) + "\n")
        logger.info(f"output path: {output_path}")


if __name__ == '__main__':
    predict()
