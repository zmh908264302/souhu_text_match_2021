import os
from datetime import datetime

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from transformers import HfArgumentParser
from transformers.optimization import AdamW

from Io.data_loader import create_batch_iter
from adversarial.adversaial import PGD
from config.arguments_utils import TrainingArguments
from evaluate.acc_f1 import evaluate
from tools.Logginger import init_logger
from tools.model_util import set_seed
from tools.plot_util import loss_acc_plot


def result_to_file(result, file_name, logger):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(args, model, eval_dataloader, device, epoch, num_epochs, mode="eval", logger=None):
    eval_preds, eval_labels = [], []
    count = 0
    eval_losses = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader, desc="evaluation")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch
            outputs = model(input_ids, input_mask, segment_ids, labels=labels, return_dict=True)

            eval_loss = outputs.loss
            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            eval_losses += eval_loss.item()
            count += 1

            eval_preds.append(outputs.logits)
            eval_labels.append(labels)

        eval_loss = eval_losses / count
        eval_predicted = torch.cat(eval_preds, dim=0).cpu()
        eval_labeled = torch.cat(eval_labels, dim=0).cpu()

        acc, prf = evaluate(eval_predicted, eval_labeled)

        logger.info(f"""\nEpoch {epoch + 1}/{num_epochs} - 
                                {mode}_loss: {eval_loss:.4f} -
                                {mode}_all_acc: {acc:.4f} - 
                                {mode}_f1:{prf[2]:.4f} \n""")

    return {
        f"{mode}_loss": eval_loss,
        f"{mode}_acc": acc,
        f"{mode}_f1": prf[2]
    }


def main():
    parser = HfArgumentParser(TrainingArguments)
    args: TrainingArguments = parser.parse_args_into_dataclasses()[0]

    # Prepare output directory
    if not args.do_eval:
        args.output_dir = os.path.join(
            args.output_dir,
            list(filter(None, args.model.strip().split("/")))[-1] + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.mkdir(args.output_dir)
    logger = init_logger("souhu-text-match-2021", args.output_dir)
    logger.info(f"Output dir: {args.output_dir}")

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1

    logger.info(f"device: {device}, n_gpu: {args.n_gpu}")

    set_seed(args)

    logger.info(f"Training arguments: {args}")

    if not args.do_eval:
        train_dataloader, num_train_steps = create_batch_iter(args, "train", logger)
    eval_dataloader, _ = create_batch_iter(args, "dev", logger)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bert_config = AutoConfig.from_pretrained(args.model, return_dict=True)
    model = BertForSequenceClassification.from_pretrained(args.model)
    model.to(device)

    if args.do_eval:
        # model.eval()
        # result = do_eval(model, eval_dataloader, device, -1, -1)
        # logger.info("***** Eval results *****")
        # for key in sorted(result.keys()):
        #     logger.info("  %s = %s", key, str(result[key]))
        pass
    else:
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        # scheduler = lr_scheduler.StepLR(optimizer, 2)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        pgd = PGD(model)
        K = 3

        # Train and evaluate
        global_step = 0
        best_dev_f1, best_epoch = float("-inf"), float("-inf")
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        train_loss2plot = []
        train_acc2plot = []
        train_f1_2plot = []
        eval_loss2plot = []
        eval_acc2plot = []
        eval_f1_2plot = []
        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            train_logits = []
            train_labels = []

            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch_ + 1} iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch

                outputs = model(input_ids, input_mask, segment_ids, labels=labels, return_dict=True)
                train_logits.append(outputs.logits)
                train_labels.append(labels)

                loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()  # 方向传播，得到正常的grad

                if args.do_adversarial:
                    # 对抗训练
                    pgd.backup_grad()

                    for t in range(K):
                        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动，first attack时备份param.data
                        if t != K - 1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        adv_outputs = model(input_ids, input_mask, segment_ids, labels=labels, return_dict=True)
                        adv_loss = adv_outputs.loss
                        if args.n_gpu > 1:
                            adv_loss = adv_loss.mean()
                        adv_loss.backward()  # 反向传播，并在正常grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数

                # 梯度下降，更新参数
                optimizer.step()
                optimizer.zero_grad()

                tr_loss += loss.item()
                global_step += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    pass

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Process = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    logger.info(f"next step learning rate = {optimizer.param_groups[0]['lr']:.8f}")

                    all_train_logits = torch.cat(train_logits, dim=0).cpu()
                    all_train_labels = torch.cat(train_labels, dim=0).cpu()
                    acc, prf = evaluate(all_train_logits, all_train_labels)

                    train_loss2plot.append(loss.item())
                    train_acc2plot.append(acc)
                    train_f1_2plot.append(prf[2])

                    loss = tr_loss / (step + 1)

                    result = do_eval(args, model, eval_dataloader, device, epoch_, args.num_train_epochs, "eval",
                                     logger)
                    scheduler.step(result["eval_loss"])
                    eval_loss2plot.append(result["eval_loss"])
                    eval_acc2plot.append(result["eval_acc"])
                    eval_f1_2plot.append((result["eval_f1"]))

                    result['global_step'] = global_step
                    result['train_loss'] = loss

                    result_to_file(result, output_eval_file, logger)

                    if args.do_eval:
                        save_model = False
                    else:
                        save_model = False
                        if result['eval_f1'] > best_dev_f1:
                            best_dev_f1 = result['eval_f1']
                            best_epoch = epoch_ + 1
                            save_model = True

                    if save_model:
                        logger.info("***** Save model *****")
                        best_model = model
                        model_to_save = model.module if hasattr(best_model, 'module') else best_model

                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        output_config_file = os.path.join(args.output_dir, "config.json")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

        logger.info(f"best epoch: {best_epoch}, best eval f1:{best_dev_f1:.4f}")

        loss_acc_plot([train_loss2plot, train_acc2plot, train_f1_2plot,
                       eval_loss2plot, eval_acc2plot, eval_f1_2plot],
                      os.path.join(args.output_dir, "loss_acc_f1.png"))
        logger.info(f"output dir: {args.output_dir}")


if __name__ == '__main__':
    main()
