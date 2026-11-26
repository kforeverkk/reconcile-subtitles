"""Resolve registered acoustic models strictly inside this tool's models folder."""
from pathlib import Path


def model_directory(root, entry):
    if not isinstance(entry, dict) or entry.get('method') not in ('character', 'word'):
        raise ValueError('模型登记需要 directory 以及 character/word method')
    root = Path(root).resolve()
    base = (root / 'models').resolve()
    if not base.is_relative_to(root):
        raise ValueError('models 目录实际指向工具目录外部')
    path = (Path(root) / entry['directory']).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError('校时模型必须位于 Gemini-Transcribe/models 目录内')
    return path


def validate_model(directory, method):
    import json
    import numpy as np
    import onnxruntime as ort
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8-sig'))
    vocab = json.loads((directory / 'vocab.json').read_text(encoding='utf-8-sig'))
    stride = config.get('conv_stride')
    if not isinstance(stride, list) or not stride or not all(type(x) is int and x>0 for x in stride):
        raise ValueError('模型缺少可用的 CTC conv_stride 配置')
    blank = config.get('pad_token_id', vocab.get('<pad>'))
    if not isinstance(blank, int) or blank not in vocab.values():
        raise ValueError('模型缺少有效的 CTC 空白符编号')
    if method == 'word' and '|' not in vocab:
        raise ValueError('按词校时模型需要 | 作为词间分隔符')
    session = ort.InferenceSession(str(directory/'model_int8.onnx'), providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    sample = np.zeros((1, 16000), dtype=np.float32)
    feed = {inputs[0].name: sample}
    for item in inputs[1:]:
        if item.name != 'attention_mask':
            raise ValueError('当前校时器不支持该 ONNX 附加输入')
        feed[item.name] = np.ones(sample.shape, dtype=np.int64)
    result = session.run(None, feed)[0]
    if result.ndim != 3 or result.shape[-1] != len(vocab):
        raise ValueError('ONNX 输出与 CTC 字表不匹配')
    return dict(vocabulary=len(vocab), blank=blank, stride=int(np.prod(stride)))
