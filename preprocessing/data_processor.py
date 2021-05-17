import json
import warnings
from itertools import islice
from typing import List

from tqdm import tqdm


def sent_label_split(line):
    """
    句子处理成单词
    :param line: 原始行
    :return: 单词， 标签
    """
    line = line.strip('\n').split('\t')
    text_a = line[0].replace(" ", "")
    text_b = line[1].replace(" ", "")
    label = line[2].replace(" ", "")
    return text_a, text_b, label


class InputExample(object):
    def __init__(self, guid, source, target, label):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
        """
        self.guid = guid
        self.text_a = source
        self.text_b = target
        self.label = label


class InputTestExample(object):
    def __init__(self, guid, source, target, id):
        """ 创建一个测试实例 """
        self.guid = guid
        self.text_a = source
        self.text_b = target
        self.id = id


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_train_examples(self, data_dir):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """读tsv文件，这里可以自定义，可以读json，csv等多种格式"""
        with open(input_file, "r", encoding='utf-8') as fr:
            lines = json.load(fr)
            return lines

    @classmethod
    def _read_csv(cls, data_path):
        with open(data_path, "r", encoding="utf-8") as fr:
            lines = []
            for line in islice(fr, 1, None):
                lines.append(line.replace("\n", "").split("\t"))
            return lines


class MyPro(DataProcessor):
    """将数据构造成example格式"""

    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"
            source = line[0]
            target = line[1]
            label = int(line[2]) if set_type != "test" else line[2]
            example = InputExample(guid=guid, source=source, target=target, label=label)
            examples.append(example)
        return examples

    def get_examples(self, data_path, file_format="json", mode="train"):
        if file_format == "json":
            lines = self._read_json(data_path)
            examples = self._create_example(lines, mode)
        elif file_format == "csv":
            lines = self._read_csv(data_path)
            examples = self._create_example(lines, mode)
        else:
            raise ValueError(f"file_format: {file_format}, don't support...")
        return examples

    def get_dev_examples(self, data_path, file_format="json"):
        warnings.warn(f"function get_dev_examples is deprecated.")
        if file_format == "json":
            lines = self._read_json(data_path)
            examples = self._create_example(lines, "dev")
        elif file_format == "de":
            lines = self._read_csv(data_path)
        else:
            raise ValueError(f"file_format: {file_format}, don't support...")
        examples = self._create_example(lines, "dev")
        return examples


def convert_examples_to_features(examples: List[InputExample], max_seq_length, tokenizer, logger=None):
    features = []
    for ex_index, example in enumerate(tqdm(examples, desc="convert to feature", ascii=True)):
        label = example.label
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)  # 最后一个加1为什么？因为使用的tokens_b，看上一行代码加了[SEP]

        # 词转换成数字
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # --------------看结果是否合理-------------------------
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(f"label: {example.label}")
        # ----------------------------------------------------

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label=label)
        features.append(feature)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
