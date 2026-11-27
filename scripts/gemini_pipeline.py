"""Run the tested Gemini stages using a prepared, shared PCM track."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave
from runtime_env import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
configure_runtime(ROOT)
os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + str(ROOT / 'bin')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import gemini_transcribe as g
import postprocess as pp
from acoustic_align import align_japanese, align_spanish
from media_audio import job_lock
from model_registry import model_directory, validate_model


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_srt(path):
    text = path.read_text(encoding='utf-8-sig').strip()
    import re
    cues = []
    for block in re.split(r'\r?\n\s*\r?\n', text):
        if not block.strip():
            continue
        lines = block.splitlines()
        start, end = lines[1].split(' --> ')
        cues.append(g.Cue(g.millis(start.replace(',', '.')), g.millis(end.replace(',', '.')), '\n'.join(lines[2:])))
    return cues


def save_srt(path, cues, offset=0, width=None):
    temporary = path.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    shifted = [replace(c, start=c.start+offset, end=c.end+offset) for c in cues]
    g.write_srt(temporary, shifted, (lambda text: pp.display_wrap(text, width)) if width else None)
    temporary.replace(path)


def check(language, config_path):
    import numpy
    import onnxruntime as ort
    from faster_whisper.vad import get_vad_model
    for executable in ('ffmpeg', 'ffprobe'):
        subprocess.run([executable, '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    get_vad_model()
    if not (ROOT / 'alignment-models.json').is_file():
        raise ValueError('请先使用 install_alignment_model.py 安装校时模型')
    models = json.loads((ROOT / 'alignment-models.json').read_text(encoding='utf-8'))
    languages = [language] if language else list(models)
    for code in languages:
        if code not in models:
            raise ValueError(f'尚未配置 {code} 的本地校时模型，请先配置 alignment-models.json')
        model = model_directory(ROOT, models[code])
        validate_model(model, models[code]['method'])
    config = g.read_config(config_path)
    print(json.dumps(dict(root=str(ROOT), python=sys.executable, base_runtime=sys.base_prefix,
                          alignment_languages=languages, api_key_configured=bool(os.environ.get('GEMINI_API_KEY') or config['api_key']),
                          api_host=g.urlsplit(g.endpoint(config['api_base_url'])).hostname,
                          cache_paths=configure_runtime(ROOT),
                          numpy=numpy.__version__), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Gemini 独立听写、整理、本地校时及歌词标记')
    parser.add_argument('audio', type=Path, nargs='?')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--language', default='')
    parser.add_argument('--config', type=Path, default=ROOT / 'gemini-config.json')
    parser.add_argument('--timeline-offset', type=float, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.check:
        check(args.language, args.config)
        return 0
    if not args.audio or not args.output_dir or not args.language:
        parser.error('需要公共音轨、--output-dir 和 --language')
    audio = args.audio.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with job_lock(out / 'run.lock'):
        stage_dir = out / '.stages'
        work = out / 'work'
        stage_dir.mkdir(exist_ok=True)
        work.mkdir(exist_ok=True)
        configure_runtime(ROOT, work / 'temp')
        with wave.open(str(audio), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError('公共音轨必须为16kHz单声道PCM16 WAV')
            duration = wav.getnframes() // 16
        config = g.read_config(args.config)
        models = json.loads((ROOT / 'alignment-models.json').read_text(encoding='utf-8'))
        if args.language not in models:
            raise ValueError(f'尚未配置 {args.language} 的本地校时模型')
        model_info = models[args.language]
        model = model_directory(ROOT, model_info)
        if not (model / 'model_int8.onnx').is_file():
            raise ValueError('本地校时模型文件缺失')
        stat = audio.stat()
        identity = dict(audio=str(audio), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                        language=args.language, timeline_offset=args.timeline_offset,
                        config={k:v for k,v in config.items() if 'key' not in k and k!='concurrency'}, model=str(model),
                        alignment_method=model_info['method'])
        identity_path = out / 'input.json'
        if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('已有任务的音轨或参数不同，请建立新任务目录')
        write_json(identity_path, identity)
        if args.dry_run:
            return g.main([str(audio), '--prepared-audio', '--work-dir', str(work), '--config', str(args.config),
                           '--language', args.language, '--raw-only', '--output', str(stage_dir/'dry-run.srt'), '--dry-run'])
        state_path = stage_dir / 'state.json'
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        if 'raw' not in state:
            attempt = 1
            raw_path = stage_dir / 'raw.srt'
            while raw_path.exists() or raw_path.with_suffix('.report.json').exists():
                attempt += 1
                raw_path = stage_dir / f'raw-{attempt}.srt'
            result = g.main([str(audio), '--prepared-audio', '--work-dir', str(work), '--config', str(args.config),
                             '--language', args.language, '--raw-only', '--output', str(raw_path)])
            if result:
                return result
            raw_report = json.loads(raw_path.with_suffix('.report.json').read_text(encoding='utf-8'))
            if raw_report['is_partial']:
                raise RuntimeError('听写存在未完成窗口，保留检查点以便恢复')
            state['raw'] = raw_path.name
            write_json(state_path, state)
        raw = load_srt(stage_dir / state['raw'])
        key = (os.environ.get('GEMINI_API_KEY') or config['api_key']).strip()
        address, text_key, model_name = g.text_target(config, g.endpoint(config['api_base_url']), key)
        ask = g.text_client(address, text_key, model_name, config)
        width = config['max_chars_per_line']
        refined_path = stage_dir / 'refined.srt'
        if refined_path.exists() and 'refinement' in state:
            refined = load_srt(refined_path)
        else:
            if config['refine_enabled']:
                refined, state['refinement'] = pp.refine(raw, ask, max_chars=width, max_duration=config['subtitle_max_seconds']*1000)
            else:
                refined = raw
                state['refinement'] = dict(disabled=True)
            save_srt(refined_path, refined)
            write_json(state_path, state)
        aligned_path = stage_dir / 'aligned.srt'
        timing_path = stage_dir / 'timing.json'
        if aligned_path.exists() and timing_path.exists():
            aligned = load_srt(aligned_path)
            timing = json.loads(timing_path.read_text(encoding='utf-8'))
        else:
            align = align_japanese if model_info['method'] == 'character' else align_spanish
            aligned, timing = align(audio, refined, duration, model, width)
            save_srt(aligned_path, aligned)
            write_json(timing_path, timing)
        final_local = stage_dir / 'marked.srt'
        if final_local.exists() and 'lyrics' in state:
            marked = load_srt(final_local)
        else:
            if config['mark_lyrics']:
                marked, state['lyrics'] = pp.mark_lyrics(aligned, ask)
            else:
                marked = aligned
                state['lyrics'] = dict(disabled=True)
            if [(c.start,c.end) for c in marked] != [(c.start,c.end) for c in aligned]:
                raise ValueError('歌词标记改变了时间码')
            save_srt(final_local, marked)
            write_json(state_path, state)
        if any(not 0 <= c.start < c.end <= duration for c in marked):
            raise ValueError('成稿存在无效时间码')
        if any(a.end>b.start for a,b in zip(marked,marked[1:])):
            raise ValueError('成稿存在时间重叠')
        offset = round(args.timeline_offset * 1000)
        for name, cues in [('raw',raw),('refined',refined),('aligned',aligned),('processed',marked)]:
            save_srt(out / (name+'.srt'), cues, offset, width if name=='processed' else None)
        write_json(out / 'timing.json', dict(timeline_offset_ms=offset, **timing))
        write_json(out / 'report.json', dict(complete=True, language=args.language, duration_ms=duration,
                   timeline_offset_ms=offset, raw_count=len(raw), refined_count=len(refined), aligned_count=len(aligned),
                   final_count=len(marked), refinement=state.get('refinement'), lyrics=state.get('lyrics'),
                   timing={k:v for k,v in timing.items() if k!='changes'}))
        print(f'Gemini 成稿已写入 {out / "processed.srt"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        message = '配置或恢复记录格式无效' if isinstance(error, json.JSONDecodeError) else str(error)
        print(message, file=sys.stderr)
        raise SystemExit(1)
