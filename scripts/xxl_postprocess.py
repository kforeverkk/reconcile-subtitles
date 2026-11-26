"""Conservative word-boundary and waveform processing; no text-model calls."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
import unicodedata
import wave

STAMP = re.compile(r'(\d+):(\d\d):(\d\d)[,.](\d{3})')
CJK = re.compile(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]')


@dataclass(frozen=True)
class Cue:
    start: int
    end: int
    text: str
    sources: tuple[int, ...]


def ms(text):
    match = STAMP.fullmatch(text.strip())
    if not match:
        raise ValueError('无效 SRT 时间码')
    h, m, s, n = map(int, match.groups())
    if m >= 60 or s >= 60:
        raise ValueError('无效 SRT 分秒')
    return ((h * 60 + m) * 60 + s) * 1000 + n


def clock(value):
    s, n = divmod(value, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f'{h:02}:{m:02}:{s:02},{n:03}'


def load_srt(path):
    text = path.read_text(encoding='utf-8-sig').strip()
    result = []
    if not text:
        return result
    for i, block in enumerate(re.split(r'\r?\n\s*\r?\n', text), 1):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].isdigit():
            raise ValueError('无法读取引擎 SRT 条目')
        a, b = lines[1].split(' --> ')
        result.append(Cue(ms(a), ms(b), '\n'.join(lines[2:]), (i,)))
    return result


def normal(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if unicodedata.category(c)[0] in 'LN')


def join(a, b):
    if not a or not b:
        return a + b
    if CJK.search(a[-1]) and CJK.search(b[0]):
        return a + b
    return a + ' ' + b


def wrap(text, width):
    if '\n' in text or len(text) <= width:
        return text
    points = [i for i in range(1, len(text)) if text[i].isspace() or text[i - 1] in '、。，,.!?！？']
    if points:
        cut = min(points, key=lambda i: abs(i - len(text) / 2))
    elif CJK.search(text):
        cut = len(text) // 2
    else:
        return text
    return text[:cut].rstrip() + '\n' + text[cut:].lstrip()


def write_srt(path, cues, width=None):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for n, c in enumerate(cues, 1):
            body = wrap(c.text, width) if width else c.text
            stream.write(f'{n}\n{clock(c.start)} --> {clock(c.end)}\n{body}\n\n')


def get_words(data):
    result = []
    for segment in data.get('segments', []):
        for word in segment.get('words') or []:
            try:
                a, b = float(word['start']), float(word['end'])
                text = str(word.get('word', word.get('text', '')))
                if math.isfinite(a) and math.isfinite(b) and 0 <= a <= b and text.strip():
                    result.append((round(a * 1000), round(b * 1000), text))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(result, key=lambda w: (w[0], w[1]))


def matching_words(cue, words):
    selected = [w for w in words if w[0] >= cue.start - 180 and w[1] <= cue.end + 180]
    # Engine SRT and JSON may be grouped differently: only accept an exact text span.
    wanted = normal(cue.text)
    if not wanted:
        return []
    for a in range(len(selected)):
        text = ''
        for b in range(a, len(selected)):
            text += normal(selected[b][2])
            if text == wanted:
                found = selected[a:b + 1]
                if all(x[1] <= y[0] + 80 for x, y in zip(found, found[1:])):
                    return found
            if len(text) >= len(wanted):
                break
    return []


def waveform_start(source, cue, previous_end, words, settings):
    import numpy as np
    search = settings['start_search_ms']
    low = max(0, previous_end, cue.start - search)
    high = min(cue.end, cue.start + search + 120)
    if high - low < 100:
        return cue
    # Prefer an actual first-word boundary; waveform alone cannot label music as speech.
    if not words or abs(words[0][0] - cue.start) > search:
        return cue
    source.setpos(low * 16)
    samples = np.frombuffer(source.readframes((high - low) * 16), dtype='<i2').astype(np.float32) / 32768
    count = len(samples) // 320
    if count < 5:
        return cue
    levels = 20 * np.log10(np.maximum(np.sqrt(np.mean(samples[:count * 320].reshape(count, 320) ** 2, axis=1)), 1e-7))
    floor = float(np.percentile(levels, 90))
    if floor < -55 or np.percentile(levels, 90) - np.percentile(levels, 10) < 10:
        return cue
    active = levels >= floor - 12
    onsets = [i for i in range(1, count - 3) if not active[i - 1] and active[i] and sum(active[i:i + 4]) >= 3]
    if not onsets:
        return cue
    actual = min((low + i * 20 for i in onsets), key=lambda t: abs(t - words[0][0]))
    candidate = max(previous_end, actual - settings['start_lead_ms'])
    if abs(candidate - cue.start) > search or abs(actual - words[0][0]) > 160 or cue.end - candidate < 600:
        return cue
    return replace(cue, start=candidate)


def process(cues, data, audio, cfg):
    for key in ('line_width', 'max_cue_seconds', 'min_display_ms', 'start_search_ms', 'start_lead_ms', 'merge_gap_ms'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'后处理参数 {key} 必须为非负整数')
    if cfg['line_width'] < 10 or cfg['max_cue_seconds'] < 1:
        raise ValueError('字幕长度或时长上限过小')
    for key in ('adjust_starts', 'merge_short_cues', 'split_long_cues', 'fix_short_duration'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} 必须为 true/false')
    changes, warnings = [], []
    def record(kind, old, new):
        if old != new:
            changes.append(dict(operation=kind, source_cues=list(old.sources),
                                before=[old.start, old.end], after=[[c.start, c.end] for c in new] if isinstance(new, list) else [new.start, new.end]))
    with wave.open(str(audio), 'rb') as source:
        end = source.getnframes() // 16
        fixed = []
        ordered = sorted(cues, key=lambda c: (c.start, c.end))
        words = get_words(data)
        for pos, c in enumerate(ordered):
            if not c.text.strip():
                warnings.append(dict(source_cues=list(c.sources), reason='空文字，未保留'))
                continue
            if c.start == c.end and 0 <= c.start <= end:
                # Some XXL versions emit a zero-length cue for a very short utterance
                # (or repeat the same hallucinated cue several times). Keep its text,
                # but make the smallest conservative legal interval and record it.
                group = [n for n in ordered if n.start == c.start and n.end == c.end]
                rank = sum(1 for n in ordered[:pos] if n.start == c.start and n.end == c.end)
                next_start = next((n.start for n in ordered[pos + 1:] if n.start > c.start), end)
                available = max(0, next_start - c.start)
                span = min(cfg['min_display_ms'], available // max(1, len(group)))
                if span > 0:
                    reason = ('同时间点重复零时长条目，按文本顺序在后续空隙内分配短时长'
                              if len(group) > 1 else
                              '零时长条目，利用后续空隙补最小时长' if available >= cfg['min_display_ms'] else
                              '零时长条目，空隙不足，使用剩余空隙')
                    c = replace(c, start=c.start + rank * span)
                else:
                    span = 1 if c.start < end else 0
                    reason = '零时长条目无可用空隙，使用 1ms 保留并需复核'
                    if span:
                        c = replace(c, start=c.start + rank * span)
                if span <= 0:
                    warnings.append(dict(source_cues=list(c.sources), reason=reason + '；无法生成正时长'))
                    continue
                amended = replace(c, end=min(end, c.start + span))
                warnings.append(dict(source_cues=list(c.sources), reason=reason))
                record('保守修复零时长', c, amended)
                c = amended
            if not 0 <= c.start < c.end <= end:
                raise ValueError(f'原字幕时间无效：{c.sources}；不猜测修补，保留原始输出供检查')
            if fixed and c.start < fixed[-1].end:
                prev = fixed[-1]
                if c.start <= prev.start or prev.end - c.start > 200:
                    warnings.append(dict(source_cues=list(prev.sources + c.sources), reason='重叠可能是同时对白，原样保留'))
                else:
                    amended = replace(prev, end=c.start)
                    record('轻微重叠裁尾', prev, amended)
                    fixed[-1] = amended
            fixed.append(c)
        aligned = []
        for c in fixed:
            evidence = matching_words(c, words)
            if cfg['adjust_starts']:
                prev_end = aligned[-1].end if aligned else 0
                new = waveform_start(source, c, prev_end, evidence, cfg)
                record('词级与波形辅助起点', c, new)
                c = new
            pieces = [c]
            if cfg['split_long_cues'] and evidence and (len(c.text) > cfg['line_width'] * 2 or c.end - c.start > cfg['max_cue_seconds'] * 1000):
                groups, current = [], []
                for word in evidence:
                    current.append(word)
                    body = ''.join(w[2] for w in current)
                    if len(current) > 1 and (len(body) >= cfg['line_width'] or word[1] - current[0][0] >= cfg['max_cue_seconds'] * 1000) and re.search(r'[\s、。，,.!?！？]$', word[2]):
                        groups.append(current)
                        current = []
                if current:
                    groups.append(current)
                proposed = [replace(c, start=max(c.start, g[0][0] - 60), end=min(c.end, g[-1][1] + 120), text=''.join(w[2] for w in g).strip()) for g in groups]
                if len(proposed) > 1 and normal(''.join(p.text for p in proposed)) == normal(c.text) and all(p.end - p.start >= 600 for p in proposed) and all(a.end <= b.start for a, b in zip(proposed, proposed[1:])):
                    pieces = proposed
                    record('按词级证据拆分', c, pieces)
                else:
                    warnings.append(dict(source_cues=list(c.sources), reason='长字幕没有足够可靠的词级切点，保留'))
            aligned.extend(pieces)
    merged = []
    for c in aligned:
        if cfg['merge_short_cues'] and merged:
            prev = merged[-1]
            text = join(prev.text, c.text)
            fragment = not re.search(r'[。！？!?…]$', prev.text.rstrip()) and (len(normal(prev.text)) <= 4 or len(normal(c.text)) <= 4)
            # Short independent responses are not automatically treated as fragments.
            response = normal(c.text) in {'はい', 'いいえ', 'うん', 'そう', 'え', 'si', 'sí', 'no', 'yes', 'ok'}
            if fragment and not response and 0 <= c.start - prev.end <= cfg['merge_gap_ms'] and len(text) <= cfg['line_width'] * 2 and c.end - prev.start <= cfg['max_cue_seconds'] * 1000:
                new = replace(prev, end=c.end, text=text, sources=tuple(dict.fromkeys(prev.sources + c.sources)))
                record('相邻残句合并', prev, new)
                merged[-1] = new
                continue
        merged.append(c)
    final = []
    for i, c in enumerate(merged):
        bound = merged[i + 1].start if i + 1 < len(merged) else end
        if cfg['fix_short_duration'] and c.end - c.start < cfg['min_display_ms'] and bound > c.end:
            new = replace(c, end=min(bound, c.start + cfg['min_display_ms']))
            record('利用句后空隙补显示时长', c, new)
            c = new
        final.append(c)
    if normal(''.join(c.text for c in fixed)) != normal(''.join(c.text for c in final)):
        raise RuntimeError('后处理文字覆盖检查失败；未写出成稿')
    for a, b in zip(final, final[1:]):
        if a.end > b.start:
            warnings.append(dict(source_cues=list(a.sources + b.sources), reason='成稿保留的原始重叠，后续复核'))
    return final, dict(changes=changes, warnings=warnings,
                       output_sources=[dict(cue=i, sources=list(c.sources)) for i, c in enumerate(final, 1)])
