from transformers import AutoConfig

import config.args as args
from Io.data_loader import create_batch_iter
from net.longformer_classify import LongformerQA
from train.train import fit
from tools.porgress_util import ProgressBar
from tools.model_util import set_seed
import torch

def start():
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    # produce_data()
    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("dev")

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)

    config = AutoConfig.from_pretrained(args.bert_model)

    model = LongformerQA.from_pretrained(args.bert_model, config=config)

    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)
