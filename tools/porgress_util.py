"""进度条"""

import sys


class ProgressBar(object):
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, epoch_size, batch_size, max_arrow=10):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.max_steps = round(epoch_size / batch_size)  # 总共处理次数 = round(epoch/batch_size)
        self.max_arrow = max_arrow  # 进度条的长度

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, train_loss, train_obj_acc, train_obj_f1, train_express_acc, train_express_f1,
                     acc_star_pos, f1_start_pos, acc_end_pos, f1_end_pos, start_or_end_crt_acc, start_and_end_crt_acc,
                     used_time, i):
        num_arrow = int(i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        num_steps = self.batch_size * i  # 当前处理数据条数
        epoch_size = self.epoch_size
        process_bar = f"{num_steps}/{epoch_size:.0f}[{'>'*num_arrow}{'-'*num_line}]{percent:.2f}% - " \
                      f"train_loss {train_loss:.4f} - train_obj_acc:{train_obj_acc:.4f}, train_obj_f1:{train_obj_f1:.4f} - " \
                      f"train_express_acc:{train_express_acc:.4f}, train_express_f1:{train_express_f1:.4f} - " \
                      f"acc_star_pos {acc_star_pos:.4f} f1_start_pos {f1_start_pos:.4f} - " \
                      f"acc_end_pos {acc_end_pos:.4f} f1_end_pos {f1_end_pos:.4f} - " \
                      f"any_crt_acc {start_or_end_crt_acc:.4f} all_crt_acc {start_and_end_crt_acc:.4f} \r"""
        # - time: {used_time: .4f}
        sys.stdout.write(process_bar)  # 这两句打印字符到终端
        sys.stdout.flush()
