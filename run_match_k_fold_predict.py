import os

import torch
from tqdm import tqdm
from transformers import HfArgumentParser

from Io.data_loader import create_batch_iter
from config.arguments_utils import TrainingArguments
from preprocessing.data_processor import InputExample
from tools.Logginger import init_logger
from tools.model_util import set_seed, load_model

logger = init_logger("dureader2021-test", "output/logs/test/")


def predict_k_fold():
    parser = HfArgumentParser(TrainingArguments)
    args: TrainingArguments = parser.parse_args_into_dataclasses()[0]

    logger.info(f"Training arguments: {args}")

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1

    logger.info(f"device: {device}, n_gpu: {args.n_gpu}")

    set_seed(args)

    if "_" not in args.output_dir:
        args.output_dir = args.output_dir + sorted(os.listdir(args.output_dir))[-1]  # 最新一次训练结果
        print(f"model {args.output_dir} predict useed")

    test_dataloader, examples = create_batch_iter(args, "test")

    # tokenizer = AutoTokenizer.from_pretrained(args.model)

    # bert_config = AutoConfig.from_pretrained(args.model, return_dict=True)

    model = load_model(args)
    model.to(device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        test_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="test", ascii=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch

            outputs = model(input_ids, input_mask, segment_ids)
            logits = torch.max(outputs.logits, dim=1)[1]
            if device.type == "cuda":
                logits = logits.cpu().numpy().astype(int)
            else:
                logits = logits.numpy()
            test_logits.extend(logits.tolist())

        pred_dt = {}
        output_path = args.output_dir + "/test.csv"
        with open(output_path, "w", encoding="utf-8") as fw:
            i = 1
            for exp, label in zip(examples, test_logits):
                print(f"write line:{i}")
                i += 1
                exp: InputExample = exp
                _id = exp.label
                question = exp.text_a
                context = exp.text_b
                pred_dt[_id] = label
                fw.write(",".join([_id, str(label)]) + "\n")
        logger.info(f"output path: {output_path}")


if __name__ == '__main__':
    predict_k_fold()
