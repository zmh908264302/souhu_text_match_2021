import os
import random

import numpy as np
import torch
from transformers import BertForSequenceClassification

from config.arguments_utils import TrainingArguments


def set_seed(args):
    """ 设置种子 """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, optimizer, scheduler, output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


def load_model(args: TrainingArguments, device):
    # Load a trained model that you have fine-tuned
    # config = AutoConfig.from_pretrained(args.model, return_dict=True)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model_state_dict = torch.load(output_model_file, map_location=device)
    model = BertForSequenceClassification.from_pretrained(args.output_dir, state_dict=model_state_dict)

    return model
