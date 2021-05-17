import time
from tools.plot_util import loss_acc_f1_plot

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import config.args as args
from evaluate.acc_f1 import evaluate, evaluate_pos
from evaluate.loss import loss_fn
from tools.Logginger import init_logger
from tools.model_util import save_model
from tools.model_util import set_seed

logger = init_logger("torch", logging_path=args.log_path)

import warnings

warnings.filterwarnings('ignore')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def fit(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, verbose=1):
    # ------------------判断CUDA模式----------------------
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()  # 多GPU
        # n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1

    model.to(device)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    # ---------------------优化器-------------------------

    t_total = num_train_steps

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_step, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # ---------------------模型初始化----------------------
    model.zero_grad()
    set_seed(args)

    global_train_loss, global_eval_loss = [], []
    train_acc_obj_class_word, train_f1_obj_class_word = [], []
    train_acc_express_word, train_f1_express_word = [], []
    eval_acc_obj_class_word, eval_f1_obj_class_word  = [], []
    eval_acc_express_word, eval_f1_express_word = [], []

    history = {
        "train_loss": global_train_loss,
        "eval_loss": global_eval_loss,

        "train_acc_obj_class_word": train_acc_obj_class_word,
        "train_f1_obj_class_word": train_f1_obj_class_word,

        "train_acc_express_word": train_acc_express_word,
        "train_f1_express_word": train_f1_express_word,

        "eval_acc_obj_class_word": eval_acc_obj_class_word,
        "eval_f1_obj_class_word": eval_f1_obj_class_word,

        "eval_acc_express_word": eval_acc_express_word,
        "eval_f1_express_word": eval_f1_express_word
    }

    # ------------------------训练------------------------------
    start = time.time()
    best_obj_word_f1 = 0
    global_step = 0

    model.zero_grad()
    set_seed(args)
    for e in range(num_epoch):
        model.train()
        train_obj_predicts, train_obj_labels, train_express_predicts, train_express_labels = [], [], [], []
        loss_epoch = 0
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, obj_ids, express_ids, _, start_poses, end_poses = batch
            obj_classify, express_classify, start_logits, end_logits, _, _= model(input_ids, segment_ids, input_mask)
            # 预测对象类词， 真实对象类词， 预测表示词， 真实表示词， 起始位置， 结束位置， 预测起始位置，预测结束位置
            train_loss = loss_fn(obj_classify, obj_ids, express_classify, express_ids,
                                 start_poses, end_poses, start_logits, end_logits)

            if n_gpu > 1:
                train_loss = train_loss.mean()
            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.backward(train_loss)
            else:
                train_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            obj_classify = obj_classify.cpu()
            obj_ids = obj_ids.cpu()
            train_obj_acc, train_obj_prf = evaluate(obj_classify, obj_ids)

            express_classify = express_classify.cpu()
            express_ids = express_ids.cpu()
            train_express_acc, train_express_prf = evaluate(express_classify, express_ids)

            start_poses = start_poses.cpu()
            start_logits = start_logits.cpu()
            end_poses = end_poses.cpu()
            end_logits = end_logits.cpu()
            acc_start_pos, f1_start_pos, acc_end_pos, f1_end_pos, start_or_end_crt_acc, start_and_end_crt_acc = \
                evaluate_pos(start_poses, start_logits, end_poses, end_logits)
            loss_epoch += train_loss.item()
            pbar.show_process(train_loss.item(), train_obj_acc, train_obj_prf[2], train_express_acc, train_express_prf[2],
                              acc_start_pos, f1_start_pos, acc_end_pos, f1_end_pos,
                              start_or_end_crt_acc, start_and_end_crt_acc,
                              time.time() - start, step)
            if global_step % 100 == 0:
                train_obj_predicts.append(obj_classify)
                train_obj_labels.append(obj_ids)
                train_express_predicts.append(express_classify)
                train_express_labels.append(express_ids)

        train_obj_predicted = torch.cat(train_obj_predicts, dim=0).cpu()
        train_obj_labeled = torch.cat(train_obj_labels, dim=0).cpu()
        train_express_predicted = torch.cat(train_express_predicts, dim=0).cpu()
        train_express_labeled = torch.cat(train_express_labels, dim=0).cpu()
        del train_obj_predicts, train_obj_labels, train_express_predicts, train_express_labels

        all_train_obj_acc, all_train_obj_prf = evaluate(train_obj_predicted, train_obj_labeled)
        all_train_express_acc, all_train_express_prf = evaluate(train_express_predicted, train_express_labeled)
        global_train_loss.append(loss_epoch / (step + 1))
        train_acc_obj_class_word.append(all_train_obj_acc)
        train_f1_obj_class_word.append(all_train_obj_prf[2])
        train_acc_express_word.append(all_train_express_acc)
        train_f1_express_word.append(all_train_express_prf[2])
        del all_train_obj_acc, all_train_obj_prf, all_train_express_acc, all_train_express_prf

        # -----------------------验证----------------------------
        count = 0
        eval_obj_predicts, eval_obj_labels, eval_express_predicts, eval_express_labels = [], [], [], []
        eval_start_pos_preds, eval_start_pos_true = [], []
        eval_end_pos_preds, eval_end_pos_true = [], []
        eval_losses = 0

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, obj_ids, express_ids, _, start_poses, end_poses = batch
                obj_classify, express_classify, start_logits, end_logits, _, _ = model(input_ids, segment_ids, input_mask)
                # 预测对象类词， 真实对象类词， 预测表示词， 真实表示词， 起始位置， 结束位置， 预测起始位置，预测结束位置
                eval_loss = loss_fn(obj_classify, obj_ids, express_classify, express_ids,
                                    start_poses, end_poses, start_logits, end_logits)

                eval_losses += eval_loss
                count += 1

                eval_obj_predicts.append(obj_classify)
                eval_obj_labels.append(obj_ids)
                eval_express_predicts.append(express_classify)
                eval_express_labels.append(express_ids)

                eval_start_pos_preds.append(start_logits)
                eval_start_pos_true.append(start_poses)
                eval_end_pos_preds.append(end_logits)
                eval_end_pos_true.append(end_poses)

            eval_obj_predicted = torch.cat(eval_obj_predicts, dim=0).cpu()
            eval_obj_labeled = torch.cat(eval_obj_labels, dim=0).cpu()
            eval_express_predicted = torch.cat(eval_express_predicts, dim=0).cpu()
            eval_express_labeled = torch.cat(eval_express_labels, dim=0).cpu()

            eval_obj_acc, eval_obj_prf = evaluate(eval_obj_predicted, eval_obj_labeled)
            eval_express_acc, eval_express_prf = evaluate(eval_express_predicted, eval_express_labeled)

            eval_acc_obj_class_word.append(eval_obj_acc)
            eval_f1_obj_class_word.append(eval_obj_prf[2])
            eval_acc_express_word.append(eval_express_acc)
            eval_f1_express_word.append(eval_express_prf[2])

            eval_start_pos_preds = torch.cat(eval_start_pos_preds, dim=0).cpu()
            eval_start_pos_true = torch.cat(eval_start_pos_true, dim=0).cpu()
            eval_end_pos_preds = torch.cat(eval_end_pos_preds, dim=0).cpu()
            eval_end_pos_true = torch.cat(eval_end_pos_true, dim=0).cpu()

            acc_start_pos, f1_start_pos, acc_end_pos, f1_end_pos, start_or_end_crt_acc, start_and_end_crt_acc = \
                evaluate_pos(eval_start_pos_true, eval_start_pos_preds, eval_end_pos_true, eval_end_pos_preds)

            avg_eval_loss = eval_losses.item() / count
            global_eval_loss.append(avg_eval_loss)
            logger.info(f"""\nEpoch {e + 1}/{num_epoch} - eval_loss: {avg_eval_loss:.4f} - 
                        eval_obj_acc: {eval_obj_acc:.4f} eval_obj_f1:{eval_obj_prf[2]:.4f} - 
                        eval_express_acc: {eval_express_acc:.4f} eval_express_f1: {eval_express_prf[2]:.4f} - 
                        eval_acc_start_pos: {acc_start_pos:.4f} eval_f1_start_pos: {f1_start_pos:.4f} - 
                        eval_acc_end_pos: {acc_end_pos:.4f} eval_f1_end_pos: {f1_end_pos:.4f} - 
                        any_crt_acc: {start_or_end_crt_acc:.4f} all_crt_acc {start_and_end_crt_acc:.4f}\n""")
            # 保存最好的模型
            if eval_obj_prf[2] > best_obj_word_f1:
                best_obj_word_f1 = eval_obj_prf[2]
                save_model(model, optimizer, scheduler, args.output_dir)

    #         if e % verbose == 0:
    #             train_losses.append(train_loss.item())
    #             train_f1.append(best_train_f1)
    #             eval_losses.append(eval_loss.item() / count)
    #             eval_f1.append(_eval_f1)
    logger.info(f"best object class word: {best_obj_word_f1:.4f}")
    loss_acc_f1_plot(history, path=args.output_dir + "loss_acc_f1_plot.png")
