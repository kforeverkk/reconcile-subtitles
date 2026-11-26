"""Standalone XXL runner. Audio preparation is reusable by both ASR branches."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from xxl_postprocess import load_srt, process, write_srt
from runtime_env import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
os.environ['PATH'] = str(ROOT) + os.pathsep + os.environ.get('PATH', '')


def run(command, log=None, cwd=None):
    if log is not None:
        with log.open('x', encoding='utf-8') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    cwd=cwd, check=False)
    else:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', check=False)
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name} 退出码 {result.returncode}；'
                           + (f'详见 {log}' if log else result.stderr[-2000:]))
    return None if log else result.stdout


def prepare_audio(media, target, stream=None, start=0, duration=None):
    data = json.loads(run(['ffprobe', '-v', 'error', '-show_streams',
                           '-show_format', '-of', 'json', str(media)]))
    tracks = [s for s in data['streams'] if s['codec_type'] == 'audio']
    if stream is None and len(tracks) == 1:
        stream = tracks[0]['index']
    if stream is None or not any(s['index'] == stream for s in tracks):
        raise ValueError('请明确 --audio-stream，全局音轨索引可选：' + ', '.join(
            f"{s['index']} ({s.get('tags', {}).get('language', '未知')})" for s in tracks))
    # Trim after normalising the timestamp gaps; do not reset an input seek first.
    filters = 'aresample=16000:async=1:first_pts=0'
    if start or duration is not None:
        filters += f',atrim=start={start}'
        if duration is not None:
            filters += f':duration={duration}'
        filters += ',asetpts=PTS-STARTPTS'
    run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-copyts', '-start_at_zero',
         '-i', str(media), '-map', f'0:{stream}', '-vn', '-af', filters,
         '-ac', '1', '-c:a', 'pcm_s16le', str(target)])
    return dict(stream=stream, media_duration=data['format'].get('duration'),
                timeline_offset_ms=round(start * 1000), filters=filters)


def main():
    cli = argparse.ArgumentParser(description='XXL 听写与词级、波形辅助后处理')
    cli.add_argument('media', type=Path)
    cli.add_argument('--config', type=Path, default=ROOT / 'config/western.json')
    cli.add_argument('--engine', type=Path, required=True)
    cli.add_argument('--models', type=Path, required=True)
    cli.add_argument('--output-dir', type=Path, required=True, help='必须是尚不存在的独立目录')
    cli.add_argument('--language', required=True, help='如 ja / es')
    cli.add_argument('--audio-stream', type=int)
    cli.add_argument('--start', type=float, default=0)
    cli.add_argument('--duration', type=float)
    cli.add_argument('--prepared-audio', action='store_true', help='使用已有 16kHz 单声道 PCM WAV，不重新抽轨')
    cli.add_argument('--timeline-offset', type=float, default=0, help='已有音频对应的原视频起点（秒）')
    args = cli.parse_args()
    import math
    for name in ('start', 'timeline_offset', 'duration'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} 必须为非负有限数')
    if args.duration == 0:
        raise ValueError('duration 必须大于零')
    if args.prepared_audio and (args.start or args.duration is not None):
        raise ValueError('已有公共音频不能在这里再次裁剪；用 --timeline-offset 标明其起点')
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    engine = args.engine.resolve()
    models = args.models.resolve()
    if not engine.is_file() or not args.media.is_file():
        raise ValueError('引擎或输入文件不存在')
    if not (models / ('faster-whisper-' + cfg['model']) / 'model.bin').is_file():
        raise ValueError('本地模型不存在；试验模块不自动下载模型')
    extras = cfg.get('extra_args', [])
    if not isinstance(extras, list) or not all(isinstance(s, str) for s in extras):
        raise ValueError('extra_args 必须是字符串数组')
    reserved = {'--model', '--model_dir', '--output_dir', '-o', '--output_format', '-f',
                '--language', '-l', '--task', '--word_timestamps', '-wt', '--clip_timestamps', '-clip'}
    if any(x.split('=')[0] in reserved for x in extras):
        raise ValueError('extra_args 不能覆盖模型、输出位置、任务、语言、词级时间或剪辑范围')
    preset = cfg.get('subtitle_preset', 'standard')
    if preset not in ('standard', 'standard_asia') or any(x in ('--standard', '--standard_asia') for x in extras):
        raise ValueError('subtitle_preset 必须单选 standard 或 standard_asia；不要在 extra_args 重复设置')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    configure_runtime(ROOT, out / 'temp')
    if args.prepared_audio:
        audio = args.media.resolve()
        preparation = dict(timeline_offset_ms=round(args.timeline_offset * 1000), reused_audio=True)
    else:
        audio = out / 'audio.wav'
        preparation = prepare_audio(args.media.resolve(), audio, args.audio_stream,
                                    args.start, args.duration)
    import wave
    with wave.open(str(audio), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError('公共音频必须为 16kHz、单声道、16位 PCM WAV')
    raw = out / 'engine'
    raw.mkdir()
    command = [str(engine), str(audio), '--model', cfg['model'], '--model_dir', str(models),
               '--device', cfg['device'], '--compute_type', cfg['compute_type'],
               '--language', args.language, '--task', 'transcribe', '--word_timestamps', 'True',
               '--vad_filter', 'True', '--vad_method', cfg['vad_method'],
               '--vad_device', cfg['vad_device'], '--' + preset, '--beep_off',
               '--output_dir', str(raw), '--output_format', 'srt', 'json', *extras]
    manifest = dict(preparation=preparation, command=command, config=cfg)
    (out / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('开始 XXL 听写；日志：', out / 'engine.log', flush=True)
    run(command, log=out / 'engine.log', cwd=str(engine.parent))
    raw_srt, raw_json = raw / (audio.stem + '.srt'), raw / (audio.stem + '.json')
    if not raw_srt.is_file() or not raw_json.is_file():
        raise RuntimeError('引擎未生成预期 SRT / JSON；保留日志和全部原始文件')
    cues = load_srt(raw_srt)
    data = json.loads(raw_json.read_text(encoding='utf-8-sig'))
    final, report = process(cues, data, audio, cfg['postprocess'])
    offset = preparation['timeline_offset_ms']
    from dataclasses import replace
    write_srt(out / 'raw.srt', [replace(c, start=c.start + offset, end=c.end + offset) for c in cues])
    write_srt(out / 'processed.srt', [replace(c, start=c.start + offset, end=c.end + offset) for c in final],
              cfg['postprocess']['line_width'])
    report.update(timeline_offset_ms=offset, raw_count=len(cues), processed_count=len(final))
    (out / 'postprocess.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成：', out / 'processed.srt')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print('错误：', exc, file=sys.stderr)
        sys.exit(1)
