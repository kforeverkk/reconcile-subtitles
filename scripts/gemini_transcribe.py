"""Transcribe one media file with Gemini. Requires FFmpeg and faster-whisper's Silero VAD."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import wave
from runtime_env import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
configure_runtime(ROOT)

RATE = 16000
GOOGLE = 'https://generativelanguage.googleapis.com/v1beta/openai/'
MAX_BODY = 18 * 1024 * 1024
TS = r'(?:\d{1,3}:){0,2}\d+(?:[.,]\d{1,3})?'
LINE = re.compile(rf'^\s*[-*•]?\s*[\[(]?\s*({TS})\s*(?:-{{1,2}}>|→|–|—|~)\s*({TS})'
                  r'\s*[\])]?\s*[:：]?\s*(.*)$')
LANGUAGE_TAG = re.compile(r'^#?\s*lang\s*[:：]\s*[A-Za-z]{2,3}\s*$', re.I)


@dataclass(frozen=True)
class Cue:
    start: int
    end: int
    text: str


@dataclass
class WindowResult:
    start: int
    end: int
    status: str
    cues: list[Cue]
    detail: str = ''
    automatic_start_adjustments: int = 0


class BadReply(ValueError):
    pass


class EndpointError(RuntimeError):
    pass


class ContentDeclined(RuntimeError):
    pass


def run(*command, data=None):
    try:
        result = subprocess.run(command, input=data, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f'找不到 {command[0]}，请确认已安装并在 PATH 中。') from exc
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace').strip() or f'{command[0]} failed')
    return result.stdout


def tracks(media):
    data = run('ffprobe', '-v', 'error', '-select_streams', 'a',
               '-show_entries', 'stream=index,codec_name,channels,start_time:stream_tags=language,title',
               '-of', 'json', str(media))
    return json.loads(data).get('streams', [])


def extract(media, wav_path, stream, until_ms=None):
    bound = [] if until_ms is None else ['-t', seconds(until_ms)]
    run('ffmpeg', '-nostdin', '-v', 'error', '-copyts', '-start_at_zero',
        '-i', str(media), '-map', f'0:{stream}', '-vn',
        '-af', 'aresample=16000:async=1:first_pts=0', '-ac', '1', '-c:a', 'pcm_s16le',
        *bound, str(wav_path))


def pcm(wav_path, start, end):
    with wave.open(str(wav_path), 'rb') as source:
        first, last = start * 16, end * 16
        source.setpos(first)
        return source.readframes(last - first)


def silero(wav_path, start, end, search_ms, diagnostic_ms, min_speech_ms):
    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError('缺少 Silero VAD。请先安装附带的 requirements.txt。') from exc
    audio = np.frombuffer(pcm(wav_path, start, end), dtype='<i2').astype(np.float32)
    audio *= 1 / 32768

    def detect(silence_ms, speech_ms):
        options = VadOptions(threshold=0.35, min_speech_duration_ms=speech_ms,
                             min_silence_duration_ms=silence_ms, speech_pad_ms=0)
        return [(start + (item['start'] // 16), min(end, start + (item['end'] // 16)))
                for item in get_speech_timestamps(audio, options)]

    return detect(search_ms, 100), detect(diagnostic_ms, min_speech_ms)


def quiet_gaps(start, end, speech):
    cursor = start
    out = []
    for a, b in speech:
        if a - cursor >= 350:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if end - cursor >= 350:
        out.append((cursor, end))
    return out


def windows(start, end, target, quiet):
    lower = max(30000, round(target * .4))
    upper = max(lower + 1000, min(round(target * 1.4), 420000))
    out = []
    while end - start > upper:
        candidates = [(b - a, (a + b) // 2) for a, b in quiet
                      if start + lower <= (a + b) // 2 <= start + upper]
        cut = (max(candidates, key=lambda x: (round(x[0] / 100), -abs(x[1] - start - target)))[1]
               if candidates else start + upper)
        out.append((start, cut))
        start = cut
    return out + ([(start, end)] if end > start else [])


def millis(value):
    fields = value.replace(',', '.').split(':')
    whole, _, ms = fields[-1].partition('.')
    if len(fields)>1 and (int(whole) >= 60 or (len(fields) == 3 and int(fields[-2]) >= 60)):
        raise BadReply('时间码字段超出范围')
    total = 0
    for value in [*fields[:-1],whole]:
        total = total * 60 + int(value)
    return total * 1000 + int(ms.ljust(3,'0'))


def seconds(value):
    return f'{value // 1000}.{value % 1000:03d}'


def clock(value):
    whole, ms = divmod(value, 1000)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f'{hours:02}:{minutes:02}:{secs:02},{ms:03}'


def tidy(cues):
    out = []
    for cue in sorted(cues, key=lambda c: (c.start, c.end)):
        if out and out[-1].end > cue.start:
            prev = out.pop()
            if prev.start == cue.start:
                if prev.text == cue.text:
                    out.append(Cue(cue.start, max(prev.end, cue.end), cue.text))
                else:
                    out.append(Cue(cue.start, max(prev.end, cue.end), prev.text + '\n' + cue.text))
                continue
            out.append(Cue(prev.start, cue.start, prev.text))
        out.append(cue)
    return out


def parse(reply, duration):
    lines = [line.strip() for line in reply.splitlines() if line.strip() and
             not line.strip().startswith('```') and not LANGUAGE_TAG.fullmatch(line.strip())]
    if len(lines) == 1 and ''.join(lines[0].split()).rstrip('.。').upper() in ('NOSPEECH','NONE','无'):
        return []
    cues, rejected = [], 0
    while lines:
        match = LINE.fullmatch(lines.pop(0))
        if not match:
            rejected += 1
            continue
        a, b, text = match.groups()
        # Gemini sometimes emits a clean header and the original text on the next line.
        if not text and lines and not LINE.fullmatch(lines[0]):
            text = lines.pop(0)
        if not re.search(r'[^\W_]',text):
            rejected += 1
            continue
        a, b = millis(a), millis(b)
        if b < a or a >= duration or b > duration + 1000:
            rejected += 1
            continue
        b = min(duration, max(b, a + 200))
        cues.append(Cue(a, b, text.strip()))
    if not cues or rejected > max(1, len(cues) // 20):
        raise BadReply('响应不是有效的逐句听写，或多条时间码越界')
    backwards = sum(b.start < a.start for a, b in zip(cues, cues[1:]))
    if backwards > .05 * len(cues):
        raise BadReply('多条时间码倒序')
    return tidy(cues)


def speech_ms(intervals, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def recognize(start, end, quiet, api, fmt, speech=(), limit=12):
    """Bound retries for one original window. Subwindows retain absolute coordinates."""
    count = 0

    def one(a, b):
        nonlocal count
        complaint = ''
        detail = ''
        switched = False
        current = fmt
        repeated = ()
        attempts = 2
        tries = 0
        while tries < attempts and count < limit:
            tries += 1
            count += 1
            try:
                reply = api(a, b, current, complaint)
                if not reply.strip():
                    detail = '接口返回空响应'
                    if not switched:
                        current = 'wav' if current == 'mp3' else 'mp3'
                        switched = True
                        attempts = max(attempts, tries + 1)
                    continue
                cues = parse(reply, b-a)
                if not cues:
                    disputed = speech_ms(speech, a, b)
                    return [WindowResult(a, b, 'no_speech', [],
                        f'Gemini 判定无人声；Silero 检到 {seconds(disputed)} 秒疑似语音'
                        if disputed >= 10000 else 'Gemini 判定无人声')]
                repeats = tuple((y.start, y.text) for x, y in zip(cues, cues[1:]) if x.text == y.text)
                if len(repeats) >= 5 and len(repeats) > len(cues) * .3:
                    agrees = len(repeats) == len(repeated) and all(
                        t == u and abs(x-y) <= 500 for (x,t),(y,u) in zip(repeats,repeated))
                    if not agrees:
                        repeated = repeats
                        raise BadReply('同一句反复出现，疑似生成陷环')
                tail = speech_ms(speech, a + cues[-1].end, b)
                if tail >= max(5000, speech_ms(speech, a, b) * .35):
                    raise BadReply(f'末句后仍有约 {seconds(tail)} 秒疑似语音，可能截断')
                return [WindowResult(a, b, 'ok',
                        [Cue(a + c.start, a + c.end, c.text) for c in cues],
                        f'{tries} 次请求；使用 {current}')]
            except BadReply as exc:
                complaint = detail = str(exc)
            except ContentDeclined:
                return [WindowResult(a, b, 'failed', [], '服务端拒绝转录此段')]
            except URLError:
                detail = '网络连接失败'
        if b - a > 60000 and count + 2 <= limit:
            middle = (a + b) // 2
            candidates = [(x + y) // 2 for x, y in quiet
                          if a + 30000 <= (x + y) // 2 <= b - 30000]
            cut = min(candidates, key=lambda value: abs(value - middle), default=middle)
            return one(a, cut) + one(cut, b)
        return [WindowResult(a, b, 'failed', [], detail or '请求预算用尽')]

    return one(start, end)


def endpoint(base):
    base = base.strip().rstrip('/')
    url = urlsplit(base)
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError('api_base_url 必须是 HTTPS 地址，且不能包含密钥、查询参数或片段。')
    if url.hostname == 'generativelanguage.googleapis.com' and url.path in ('', '/v1beta', '/v1beta/openai'):
        return GOOGLE + 'chat/completions'
    return base if url.path.endswith('/chat/completions') else base + '/chat/completions'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def audio_bytes(source, a, b, fmt):
    raw = pcm(source, a, b)
    if fmt == 'mp3':
        return run('ffmpeg', '-nostdin', '-v', 'error', '-f', 's16le',
                   '-ar', '16000', '-ac', '1', '-i', 'pipe:0', '-c:a', 'libmp3lame',
                   '-b:a', '64k', '-f', 'mp3', 'pipe:1', data=raw)
    import io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as sink:
        sink.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        sink.writeframes(raw)
    return buf.getvalue()


def prompt(duration, language, complaint):
    text = (
        '请逐句转写所给音频中的原语种对白。不要翻译、改写、解释或补写没听到的话。'
        '背景音乐中可听见的对白、可辨的歌词及片段边缘的半句也保留；不加音符或音效标签。'
        '按原语种补必要标点；一句跨多条时只在真正句末写句末标点。'
        '每行格式：[MM:SS.mmm --> MM:SS.mmm] 原文。'
        f'时间从这段音频的起点 0 秒算起，不超过 {seconds(duration)} 秒，按开始时间排序。'
        '每行尽量不长于 6 秒，按自然停顿切分。没有任何可辨人声时只返回 NO SPEECH。')
    if language:
        text += f'主要原声语言：{language}。'
    if complaint:
        text += f'上次结果未通过校验：{complaint}。请重新听写本片段。'
    return text


def gemini(source, url, key, config, language, chunk_dir=None):
    def ask(a, b, fmt, complaint):
        content = audio_bytes(source, a, b, fmt)
        if chunk_dir is not None:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            (chunk_dir / f'{a:010d}-{b:010d}.{fmt}').write_bytes(content)
        encoded = base64.b64encode(content).decode('ascii')
        if len(encoded) > MAX_BODY:
            raise BadReply('片段超过内联大小限制，需拆段')
        data = json.dumps({
            'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt(b - a, language, complaint)},
                {'type': 'input_audio', 'input_audio': {'data': encoded, 'format': fmt}},
            ]}],
        }).encode()
        request = Request(url, data, headers={'Authorization': f'Bearer {key}',
                                              'Content-Type': 'application/json'})
        with build_opener(NoRedirect()).open(request, timeout=240) as response:
            try:
                result = json.load(response)
            except (ValueError, UnicodeError) as exc:
                raise BadReply('接口没有返回有效 JSON') from exc
        if not isinstance(result, dict):
            raise BadReply('响应 JSON 格式不正确')
        choices = result.get('choices') or []
        if not choices:
            return ''
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise BadReply('接口返回的候选列表格式不正确')
        choice = choices[0]
        if choice.get('finish_reason') in ('content_filter', 'SAFETY', 'RECITATION'):
            raise ContentDeclined
        if choice.get('finish_reason') == 'length':
            raise BadReply('服务端报告输出被截断')
        usage = result.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        count = usage.get('prompt_tokens')
        # Usage is a heuristic; only use a clear audio-vs-text discrepancy on Google.
        if urlsplit(url).hostname == 'generativelanguage.googleapis.com' and (
                b - a >= 30000 and isinstance(count, int) and 0 < count < .8 * 25 * (b-a) / 1000):
            raise BadReply('接口记录的输入用量远低于音频请求，可能没有收到音频')
        message = choice.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        return content if isinstance(content, str) else ''

    def call(*args):
        try:
            return ask(*args)
        except HTTPError as exc:
            status = exc.code
            if status in (401, 403, 404, 400) or 300 <= status <= 399:
                hints = {400: '请检查接口是否兼容 input_audio 与 reasoning_effort；如档位不支持可手动改 low。',
                         401: '请检查 API Key。', 403: '请检查密钥或模型权限。',
                         404: '请检查 API 地址和模型名。'}
                raise EndpointError(f'Gemini HTTP {status}。{hints.get(status, "接口发生重定向。")}') from None
            if status == 429:
                raise EndpointError('Gemini HTTP 429：请检查额度，或降低并发后重试。') from None
            if status in (413, 422):
                raise BadReply(f'Gemini HTTP {status}：音频请求未被接受，需检查格式或拆短。') from None
            if status >= 500:
                time.sleep(2)
                raise BadReply(f'Gemini HTTP {status}：服务暂不可用。') from None
            raise EndpointError(f'Gemini HTTP {status}。') from None
        except TimeoutError:
            raise BadReply('接口请求超时') from None
    return call


def text_target(config, audio_url, audio_key):
    configured = config['text_api_base_url'].strip()
    url = endpoint(configured) if configured else audio_url
    key = (os.environ.get('GEMINI_TEXT_API_KEY') or config['text_api_key']).strip()
    if not key and url != audio_url:
        raise ValueError('文字接口地址与音频接口不同，请单独填写 text_api_key。')
    return url, key or audio_key, config['text_model'].strip() or config['model']


def text_client(url, key, model, config):
    blocked = ''

    def ask(system, text):
        nonlocal blocked
        if blocked:
            raise EndpointError(blocked)
        payload = dict(model=model,messages=[dict(role='system',content=system),
                                             dict(role='user',content=text)])
        if model == config['model'] and urlsplit(url).hostname == urlsplit(GOOGLE).hostname:
            payload['reasoning_effort']=config['reasoning_effort']
        request = Request(url,json.dumps(payload,ensure_ascii=False).encode(),
                          headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
        try:
            with build_opener(NoRedirect()).open(request,timeout=120) as response:
                result=json.load(response)
        except HTTPError as exc:
            message=f'文字接口 HTTP {exc.code}'
            if exc.code in (401,402,403,404,429):
                blocked=message
            raise EndpointError(message) from None
        except (OSError, ValueError):
            raise EndpointError('文字接口连接失败或响应无效') from None
        try:
            choice=result['choices'][0]
            if choice.get('finish_reason') in ('length','content_filter','SAFETY','RECITATION'):
                raise BadReply('文字接口返回未完成的响应')
            text=choice['message']['content']
            if isinstance(text,list):
                text=''.join(p.get('text','') for p in text if isinstance(p,dict))
        except (KeyError,IndexError,TypeError,AttributeError):
            raise BadReply('文字接口没有返回有效候选') from None
        if not isinstance(text,str) or not text.strip():
            raise BadReply('文字接口返回空结果')
        return text

    return ask


def snap_start(source, cue):
    """Move a too-early start toward sustained onset, never move the end."""
    import numpy as np
    data = np.frombuffer(pcm(source, cue.start, cue.end), dtype='<i2').astype(np.float32)
    count = len(data) // 320
    if count < 5:
        return cue
    frames = data[:count * 320].reshape(count, 320) / 32768
    levels = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames * frames, axis=1)), 1e-7))
    reference = float(np.percentile(levels, 90))
    if reference < -60:
        return cue
    active = levels >= reference - 12
    for i in range(min(count - 4, 76)):
        if active[i] and sum(active[i+1:i+5]) >= 2:
            new = min(cue.start + i * 20 - 100, cue.end - 500)
            return Cue(max(cue.start, new), cue.end, cue.text)
    return cue


def read_config(path):
    config = {
        'api_base_url': GOOGLE, 'api_key': '', 'model': 'gemini-3.8-flash',
        'reasoning_effort': 'none', 'window_seconds': 180, 'audio_format': 'mp3',
        'concurrency': 5, 'vad_min_speech_ms': 100, 'vad_min_silence_ms': 1500,
        'refine_enabled': True, 'mark_lyrics': True, 'max_chars_per_line':42,
        'subtitle_max_seconds':6, 'text_api_base_url':'', 'text_api_key':'', 'text_model':'',
    }
    if not path.is_file():
        raise ValueError('找不到配置文件，请保留随包的 gemini-config.json，或用 --config 指定。')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.keys() - config.keys():
        raise ValueError('配置内容或字段不正确，请使用附带的 JSON 模板。')
    config.update(data)
    for field, lo, hi in (('window_seconds', 60, 420), ('concurrency', 1, 8),
                          ('vad_min_speech_ms', 0, 5000), ('vad_min_silence_ms', 100, 10000),
                          ('max_chars_per_line',10,120), ('subtitle_max_seconds',1,15)):
        if type(config[field]) is not int or not lo <= config[field] <= hi:
            raise ValueError(f'{field} 必须为 {lo} 到 {hi} 之间的整数。')
    for field in ('api_base_url', 'api_key', 'model', 'reasoning_effort', 'audio_format',
                  'text_api_base_url','text_api_key','text_model'):
        if not isinstance(config[field], str):
            raise ValueError(f'{field} 必须为字符串。')
    if config['audio_format'] not in ('mp3', 'wav') or config['reasoning_effort'] not in ('none', 'low'):
        raise ValueError('audio_format 只接受 mp3/wav，reasoning_effort 只接受 none/low。')
    for field in ('refine_enabled','mark_lyrics'):
        if type(config[field]) is not bool:
            raise ValueError(f'{field} 必须为 true 或 false。')
    return config


def assess(results, cues, intervals):
    covered = speech_ms(intervals, results[0].start, results[-1].end)
    contains = sum(speech_ms(intervals, c.start, c.end) for c in cues)
    warnings = []
    if not covered:
        warnings.append('Silero 未检出语音，覆盖率不可用。')
    elif contains / covered < .8:
        warnings.append('有较多 Silero 可疑语音不在字幕内，请抽听确认；检测也可能被配乐误导。')
    # Diagnostic only: the VAD tail can be music, and segment boundaries are estimates.
    pairs = []
    for item in results:
        if item.cues:
            tail = max((min(item.end, b) - item.start for a,b in intervals
                        if a < item.end and b > item.start), default=0)
            if tail >= 30000:
                pairs.append((tail, tail - (item.cues[-1].end - item.start)))
    slope = round(sum(a*b for a,b in pairs) / sum(a*a for a,_ in pairs) * 1000, 1) if len(pairs) >= 4 else None
    if slope is not None and slope > 20:
        warnings.append('多个片段的末句偏早，可能截断或发生段内漂移；请抽听靠近片尾和分段接缝。')
    gaps = []
    for a, b in intervals:
        cursor = a
        for c in cues:
            if c.end <= cursor:
                continue
            if c.start >= b:
                break
            if c.start - cursor >= 2500:
                gaps.append((cursor, c.start))
            cursor = max(cursor, c.end)
        if b - cursor >= 2500:
            gaps.append((cursor, b))
    return dict(silero_speech_seconds=covered/1000,
                silero_speech_covered_by_subtitles_seconds=contains/1000,
                silero_time_coverage_percent=round(contains/covered*100,1) if covered else None,
                possible_fast_clock_ms_per_second=slope, warnings=warnings,
                uncovered_silero_intervals_2_5_seconds_or_longer=[
                    [clock(a), clock(b)] for a, b in gaps])


def process_windows(jobs, process, parallelism):
    """Stop sending new requests when a worker reports a fatal API error."""
    pool = ThreadPoolExecutor(max_workers=parallelism)
    groups, running = {}, {}
    pending = iter(enumerate(jobs))

    def submit():
        item = next(pending, None)
        if item is not None:
            index, job = item
            running[pool.submit(process, job)] = index

    try:
        for _ in range(parallelism):
            submit()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                groups[running.pop(future)] = future.result()
            for _ in finished:
                submit()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [r for index in range(len(jobs)) for r in groups[index]]


def write_srt(path, cues, wrap=None):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for n,cue in enumerate(cues,1):
            text=wrap(cue.text) if wrap else cue.text
            stream.write(f'{n}\n{clock(cue.start)} --> {clock(cue.end)}\n{text}\n\n')


def main(argv=None):
    cli = argparse.ArgumentParser(description='Gemini 原语种听写、短句整理和歌词标记。')
    cli.add_argument('media', type=Path)
    cli.add_argument('--config', type=Path, default=ROOT / 'gemini-config.json')
    cli.add_argument('--language', default='', help='如 ja；省略则保留原声语言')
    cli.add_argument('--audio-stream', type=int, help='ffprobe 的容器全局流索引')
    cli.add_argument('--start', type=float, default=0, help='试跑起点，单位秒')
    cli.add_argument('--duration', type=float, help='试跑时长，单位秒；SRT 仍用原片坐标')
    cli.add_argument('--output', type=Path, help='目标 SRT 文件，默认在D盘任务目录')
    cli.add_argument('--dry-run', action='store_true', help='实际运行 FFmpeg/Silero，但不访问网络')
    cli.add_argument('--raw-only', action='store_true', help='只交付听写初稿，跳过字幕整理和歌词识别')
    cli.add_argument('--prepared-audio', action='store_true', help='复用已抽取的16kHz单声道PCM WAV')
    cli.add_argument('--work-dir', type=Path, help='切片、窗口恢复记录和临时文件目录')
    args = cli.parse_args(argv)
    if not args.media.is_file():
        raise ValueError('找不到输入媒体。')
    import math
    if not math.isfinite(args.start) or args.start < 0 or (
            args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0)):
        raise ValueError('起点必须非负，时长必须为正。')
    from datetime import datetime
    import uuid
    default_job = ROOT.parent / 'Subtitle-Transcribe-Work' / f'gemini-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
    output = args.output or default_job / 'processed.srt'
    report_file = output.with_suffix('.report.json')
    raw_output = output if args.raw_only else output.with_suffix('.raw.srt')
    if output.suffix.lower() != '.srt':
        raise ValueError('--output 必须为 .srt。')
    if not args.dry_run and any(path.exists() for path in (output,raw_output,report_file)):
        raise ValueError('输出已存在，请用 --output 指定另外的文件名。')
    config = read_config(args.config)
    url = endpoint(config['api_base_url'])
    key = (os.environ.get('GEMINI_API_KEY') or config['api_key']).strip()
    if not args.dry_run and not key:
        raise ValueError('请在 gemini-config.json 填写 api_key，或设置 GEMINI_API_KEY。')
    if not args.dry_run and not args.raw_only and (config['refine_enabled'] or config['mark_lyrics']):
        text_endpoint,text_key,text_model = text_target(config,url,key)
    entries = tracks(args.media)
    if len(entries) == 1 and args.audio_stream is None:
        entries = [entries[0]]
    else:
        available = ', '.join(f"#{e['index']} {e.get('tags', {}).get('language', '')}" for e in entries)
        entries = [e for e in entries if e['index'] == args.audio_stream]
        if len(entries) != 1:
            raise ValueError(f'请用 --audio-stream 选择音轨。可选：{available or "没有音轨"}')
    track = entries[0]
    start = round(args.start * 1000)
    requested = None if args.duration is None else start + round(args.duration * 1000)
    work_dir = args.work_dir or output.parent / 'work'
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gemini-', dir=work_dir) as folder:
        source = args.media if args.prepared_audio else Path(folder)/'track.wav'
        print(f'提取音轨 #{track["index"]}，运行 Silero VAD…', file=sys.stderr, flush=True)
        if not args.prepared_audio:
            extract(args.media, source, track['index'], requested)
        with wave.open(str(source), 'rb') as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                raise ValueError('识别音频必须为16kHz、单声道、16位PCM WAV')
            end = audio.getnframes() // 16
        if requested is not None:
            end = min(end, requested)
        if start >= end:
            raise ValueError('试跑区间没有音频。')
        cuts, speech = silero(source, start, end, 300, config['vad_min_silence_ms'], config['vad_min_speech_ms'])
        quiet = quiet_gaps(start, end, cuts)
        jobs = windows(start, end, config['window_seconds'] * 1000, quiet)
        summary = {'audio_stream': track['index'], 'stream_start_time': track.get('start_time'),
                   'model': config['model'], 'api_host': urlsplit(url).hostname,
                   'original_windows': [[clock(a),clock(b)] for a,b in jobs],
                   'vad': {'engine': 'faster-whisper Silero', 'threshold': .35, 'cut_silence_ms': 300,
                           'diagnostic_silence_ms': config['vad_min_silence_ms'],
                           'diagnostic_min_speech_ms': config['vad_min_speech_ms'],
                           'speech_intervals': len(speech), 'speech_seconds': speech_ms(speech,start,end)/1000}}
        if args.dry_run:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        api = gemini(source, url, key, config, args.language, work_dir / 'chunks')
        print(f'向 {urlsplit(url).hostname} 发送 {len(jobs)} 个音频窗口（失败窗口可能重试）。',
              file=sys.stderr, flush=True)

        def process(job):
            a, b = job
            checkpoint = work_dir / f'window-{a}-{b}.json'
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding='utf-8'))
                restored = [WindowResult(**dict(item, cues=[Cue(**c) for c in item['cues']])) for item in saved]
                if restored and restored[0].start == a and restored[-1].end == b and all(r.status != 'failed' for r in restored):
                    return restored
            results = recognize(a, b, quiet, api, config['audio_format'], speech)
            from dataclasses import asdict
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(checkpoint)
            print(f'{clock(a)}–{clock(b)}：' + ', '.join(r.status for r in results),
                  file=sys.stderr, flush=True)
            return results

        results = process_windows(jobs, process, config['concurrency'])
        for r in results:
            updated = [snap_start(source, c) for c in r.cues]
            r.automatic_start_adjustments = sum(a.start != b.start for a, b in zip(r.cues, updated))
            r.cues = updated
        cues = tidy(c for r in results for c in r.cues)
    report = dict(summary, **assess(results, cues, speech), subtitle_count=len(cues),
                  windows=[dict(start=clock(r.start), end=clock(r.end), status=r.status,
                                cue_count=len(r.cues), detail=r.detail,
                                start_adjustments=r.automatic_start_adjustments) for r in results])
    output.parent.mkdir(parents=True, exist_ok=True)
    if cues:
        write_srt(raw_output,cues)
        print(f'已保存 {len(cues)} 条听写原稿：{raw_output}',file=sys.stderr,flush=True)

    failures=sum(r.end-r.start for r in results if r.status=='failed')
    report['raw_subtitle_count']=len(cues)
    report['unresolved_seconds']=failures/1000
    report['is_partial']=bool(failures)
    report['raw_srt']=raw_output.name if cues else None
    report['finished_srt']=None
    if cues and not args.raw_only and failures <= (end-start)*.05:
        from postprocess import display_wrap, mark_lyrics, refine, segment
        width=config['max_chars_per_line']
        max_duration=config['subtitle_max_seconds']*1000
        ask=(text_client(text_endpoint,text_key,text_model,config)
             if config['refine_enabled'] or config['mark_lyrics'] else None)
        cues=segment(cues,width,max_duration)
        stages={'segmented_count':len(cues)}
        if config['refine_enabled']:
            print('正在整理原语种字幕…',file=sys.stderr,flush=True)
            cues,stages['refine']=refine(cues,ask,max_chars=width,max_duration=max_duration)
        if config['mark_lyrics']:
            print('正在识别歌词并加音符…',file=sys.stderr,flush=True)
            cues,stages['lyrics']=mark_lyrics(cues,ask)
        write_srt(output,cues,lambda text:display_wrap(text,width))
        report['finished_srt']=output.name
        report['postprocess']=stages
        report['subtitle_count']=len(cues)
        print(f'{len(cues)} 条整理稿：{output}')
    elif cues and not args.raw_only:
        print('未通过识别的音频超过 5%，只保留部分听写原稿。',file=sys.stderr)
    with report_file.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    if not cues:
        print(f'没有可交付字幕。请查看报告：{report_file}', file=sys.stderr)
        return 2
    print(f'覆盖报告：{report_file}')
    return 2 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as error:
        # Avoid echoing malformed config contents (which may contain the key).
        message = '配置或接口响应不是有效 JSON。' if isinstance(error, json.JSONDecodeError) else str(error)
        print(f'错误：{message}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
