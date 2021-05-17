from torch.nn import CrossEntropyLoss


def ce_loss(input, target, ignored_index=None):
    if ignored_index:
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    else:
        loss_fct = CrossEntropyLoss()
    return loss_fct(input, target).mean()


def loss_fn(obj_classify, obj_ids, express_classify, express_ids,
            start_poses, end_poses, start_logits, end_logits):
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_poses.clamp_(0, ignored_index)
    end_poses.clamp_(0, ignored_index)

    loss_obj = ce_loss(obj_classify, obj_ids)
    loss_express = ce_loss(express_classify, express_ids)

    loss_start_pos = ce_loss(start_logits, start_poses, ignored_index)
    loss_end_pos = ce_loss(end_logits, end_poses, ignored_index)

    total_loss = (loss_obj + loss_express + (loss_start_pos + loss_end_pos) / 2) / 3
    return total_loss


def cross_loss_fn(obj_classify, obj_ids):
    loss_obj = ce_loss(obj_classify, obj_ids)
    return loss_obj
