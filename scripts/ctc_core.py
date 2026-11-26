"""Offline Japanese timing pilot. Requires already-transcribed SRT and local ONNX weights."""
from __future__ import annotations
from collections import defaultdict
import json
import re
import unicodedata
from pathlib import Path
import wave

import numpy as np
import onnxruntime as ort
import gemini_transcribe as g

PAD = 2800
CORE = 24000


def prepare(line, vocab):
    original = [c for c in unicodedata.normalize('NFKC', re.sub(r'<[^>]*>', '', line))
                if unicodedata.category(c)[0] in 'LN']
    labels = [vocab[c] if c in vocab else vocab[c.lower()]
              for c in original if c in vocab or c.lower() in vocab]
    return labels, len(labels)/max(1,len(original))


def forced(logp, labels, blank, expected_first, expected_last):
    """CTC Viterbi over the existing letters with free outside audio and a gentle time prior."""
    states = np.full(2*len(labels)+1, blank, dtype=np.int32)
    states[1::2] = labels
    skip = np.zeros(len(states), np.bool_)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    score = np.full(len(states), -np.inf, dtype=np.float32)
    score[0] = 0
    back = np.zeros((len(logp), len(states)), dtype=np.uint8)
    best, last_frame, s = -np.inf, -1, -1
    for p, frame in enumerate(logp):
        score[0] = -abs(p-expected_first) * .007
        stay = score + frame[states]
        prev = np.r_[-np.inf, score[:-1]] + frame[states]
        two = np.r_[-np.inf, -np.inf, score[:-2]] + frame[states]
        two[~skip] = -np.inf
        back[p] = np.argmax([stay, prev, two],axis=0)
        score = np.maximum(np.maximum(stay, prev), two)
        # Finishing in the final character allows a clean boundary from its emission.
        candidate = score[-2] - abs(p-expected_last)*.007
        if candidate > best:
            best, last_frame, s = candidate, p, len(states)-2
    if last_frame < 0:
        raise ValueError('No forced path')
    spans = [[] for _ in labels]
    for p in range(last_frame,-1,-1):
        if s == 0:
            break
        if s % 2:
            spans[s//2].append(p)
        s -= int(back[p,s])
    if any(not x for x in spans):
        raise ValueError('Incomplete forced path')
    return [(min(block),max(block),float(np.exp(max(logp[i,l] for i in block))))
            for block,l in zip(spans,labels)]


def emissions(source, a, b, session):
    sample = np.frombuffer(g.pcm(source,a,b),dtype='<i2').astype(np.float32)
    sample = (sample - sample.mean()) / np.sqrt(sample.var() + 1e-7)
    names = session.get_inputs()
    feed = {names[0].name:sample[None,:]}
    for item in names[1:]:
        if item.name != 'attention_mask':
            raise ValueError('Unknown ONNX input')
        feed[item.name] = np.ones((1,len(sample)),dtype=np.int64)
    out = session.run(None,feed)[0][0].astype(np.float32)
    stable = out - out.max(axis=1,keepdims=True)
    return stable - np.log(np.exp(stable).sum(axis=1,keepdims=True))

