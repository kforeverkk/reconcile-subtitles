"""Generate independent Hydrogen evidence for selected subtitle intervals."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
import wave

import gemini_transcribe as gemini
from media_audio import job_lock, task_output_path, workflow_config


MODEL = 'gemini-4-hydrogen'
URL = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'
LANGUAGES = '逐句辨别实际语言，混说时各用原文。保留可辨歌词和背景对白。只写确实听清的词。'
UNTIMED = (
    '请听附件，逐句写出实际听见的人声，逐行输出原语种。'
    '禁止翻译、补写、解释或添加时间戳。无可辨人声时只写 NO SPEECH。' + LANGUAGES
)
REFUSAL = re.compile(
    r'无法(?:听|访问|读取|转写|处理)|未(?:提供|收到)音频|没有(?:包含|收到|提供)音频|请.*上传.*音频|'
    r"cannot (?:access|hear|transcribe)|no audio (?:file|was provided)", re.I,
)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def make_plan(ranges, offset, duration):
    if not isinstance(ranges, list):
        raise ValueError('待核清单必须是 JSON 数组')
    seen = set()
    for item in ranges:
        if not isinstance(item, dict):
            raise ValueError('待核清单的每项必须是对象')
        label, start, end = (item.get(key) for key in ('id', 'start_ms', 'end_ms'))
        if (not isinstance(label, str) or not label.strip() or label in seen
                or type(start) is not int or type(end) is not int or not 0 <= start < end):
            raise ValueError('每项需要唯一字符串 id 和有效整数 start_ms、end_ms')
        seen.add(label)
        if start < offset or end > offset + duration:
            raise ValueError(f'待核项 {label!r} 超出所选音轨区间，请核对剪辑和偏移')
        cursor = start
        while cursor < end:
            stop = min(end, cursor + 38000)
            yield dict(id=label, start_ms=cursor, end_ms=stop,
                       clip_start_ms=max(offset, cursor - 3500),
                       clip_end_ms=min(offset + duration, stop + 3500))
            cursor = stop


def request_audio(audio, prompt, key, timed):
    attachment = {'inline_data': {'mime_type': 'audio/wav', 'data': audio}}
    instruction = {'text': prompt}
    data = json.dumps({
        'contents': [{'role': 'user', 'parts': [instruction, attachment] if timed
                      else [attachment, instruction]}],
        'generationConfig': {'temperature': 0},
    }).encode()
    request = Request(URL, data=data, headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
    with build_opener(gemini.NoRedirect()).open(request, timeout=150) as response:
        result = json.load(response)
    candidate = next(iter(result.get('candidates') or []), {})
    if candidate.get('finishReason') != 'STOP':
        raise gemini.BadReply(candidate.get('finishReason') or '没有返回候选')
    text = '\n'.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                     if not part.get('thought')).strip()
    if not text or REFUSAL.search(text):
        raise gemini.BadReply('服务没有返回可用听写')
    return text


def recognize(audio, timed_prompt, duration, key):
    errors = []
    for timed, prompt in ((True, timed_prompt), (False, UNTIMED)):
        for attempt in range(2):
            try:
                text = request_audio(audio, prompt, key, timed)
                if timed:
                    cues = [asdict(cue) for cue in gemini.parse(text, duration)]
                    status = 'timed' if cues else 'no_speech'
                else:
                    cues = []
                    status = 'no_speech' if re.fullmatch(r'NO\s*SPEECH[.。]?', text, re.I) else 'text_only'
                return dict(status=status, text=text, cues=cues, errors=errors)
            except HTTPError as error:
                errors.append(f'HTTP {error.code}')
                if error.code not in (429, 500, 502, 503, 504):
                    return dict(status='failed', errors=errors)
            except (TimeoutError, URLError) as error:
                errors.append(type(error).__name__)
            except ValueError as error:
                errors.append(type(error).__name__)
                break
            if not attempt:
                time.sleep(3)
        # The untimed prompt can recover a response that could not produce timecodes.
    return dict(status='failed', errors=errors)


def collect_clip(clip, audio_path, offset, output, key):
    start, end = clip['clip_start_ms'], clip['clip_end_ms']
    wav = gemini.audio_bytes(audio_path, start - offset, end - offset, 'wav')
    prompt = gemini.prompt(end - start, '', '') + LANGUAGES
    request_fingerprint = '\0'.join((URL, prompt, UNTIMED)).encode()
    digest = sha256(request_fingerprint + wav).hexdigest()
    wav_path = output / f'{digest}.wav'
    cache = wav_path.with_suffix('.json')
    if not wav_path.exists() or wav_path.read_bytes() != wav:
        temp = wav_path.with_suffix('.tmp')
        temp.write_bytes(wav)
        temp.replace(wav_path)
    if cache.exists():
        result = json.loads(cache.read_text(encoding='utf-8'))
    else:
        result = recognize(base64.b64encode(wav).decode('ascii'), prompt, end - start, key)
        if result['status'] != 'failed':
            write_json(cache, result)
    absolute = [dict(start_ms=start + cue['start'], end_ms=start + cue['end'], text=cue['text'])
                for cue in result.get('cues', [])]
    return {**clip, **result, 'cues': absolute, 'audio': wav_path.name,
            'evidence': cache.name if cache.exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True, help='shared/audio.wav')
    parser.add_argument('--ranges', type=Path, required=True, help='JSON array of pending ranges')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeline-offset-ms', type=int, required=True)
    parser.add_argument('--config', type=Path, default=gemini.ROOT / 'gemini-config.json')
    parser.add_argument('--dry-run', action='store_true', help='validate ranges and save a plan without API calls')
    args = parser.parse_args()
    workflow = workflow_config(gemini.ROOT)
    args.out = task_output_path(args.out, Path(workflow['work_root']))
    with wave.open(str(args.audio), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            parser.error('音轨必须是 16 kHz 单声道 16 位 PCM WAV')
        duration = source.getnframes() // 16
    ranges = json.loads(args.ranges.read_text(encoding='utf-8-sig'))
    if args.timeline_offset_ms < 0:
        parser.error('原片偏移不能为负数')
    plan = list(make_plan(ranges, args.timeline_offset_ms, duration))
    args.out.mkdir(parents=True, exist_ok=True)
    with job_lock(args.out / '.lock'):
        write_json(args.out / 'plan.json', {'model': MODEL, 'clips': plan})
        if args.dry_run:
            return
        results = []
        if plan:
            config = gemini.read_config(args.config)
            if gemini.endpoint(config['api_base_url']) != gemini.GOOGLE + 'chat/completions':
                parser.error('此入口使用 Google 官方 Gemini API，请先核对授权和已配置的接口')
            key = (os.environ.get('GEMINI_API_KEY') or config['api_key']).strip()
            if not key:
                parser.error('缺少现有 Gemini API Key')
            # Equal audio ranges share one request and one cache entry, even when rows differ.
            grouped = {(clip['clip_start_ms'], clip['clip_end_ms']): clip for clip in plan}

            # Disjoint times can still contain identical PCM. Process each range independently
            # in its own cache directory to avoid concurrent writes to the same hash.
            def process_at_time(clip):
                start, end = clip['clip_start_ms'], clip['clip_end_ms']
                folder = args.out / f'{start:010d}-{end:010d}'
                folder.mkdir(exist_ok=True)
                result = collect_clip(clip, args.audio, args.timeline_offset_ms, folder, key)
                for field in ('audio', 'evidence'):
                    if result[field]:
                        result[field] = f'{folder.name}/{result[field]}'
                return result

            with ThreadPoolExecutor(max_workers=min(4, config['concurrency'])) as pool:
                evidence = dict(zip(grouped, pool.map(process_at_time, grouped.values()), strict=True))
            for clip in plan:
                result = evidence[clip['clip_start_ms'], clip['clip_end_ms']]
                results.append({**result, **clip})
        write_json(args.out / 'review.json', {'model': MODEL, 'clips': results})
        failed = sum(result['status'] == 'failed' for result in results)
        print(f'{MODEL}: {len(results)} 个切片记录，{failed} 个请求未取得可用结果')
        return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
