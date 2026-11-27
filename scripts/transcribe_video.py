"""Prepare one track and run independent Gemini and XXL branches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from runtime_env import configure_runtime

from media_audio import prepare_audio, run, job_lock, task_output_path, workflow_config, configure_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
configure_runtime(ROOT)
os.environ['PATH'] = str(ROOT / 'bin') + os.pathsep + os.environ.get('PATH', '')


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def branch_environment(root, job):
    temp = job / 'temp'
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               TEMP=str(temp), TMP=str(temp), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(root/'cache/huggingface'), TORCH_HOME=str(root/'cache/torch'),
               XDG_CACHE_HOME=str(root/'cache'), CUDA_CACHE_PATH=str(root/'cache/cuda'),
               PATH=os.environ['PATH']+os.pathsep+str(root/'bin')+os.pathsep+str(root))
    for name, relative in {
        'HF_HUB_CACHE':'huggingface/hub', 'HUGGINGFACE_HUB_CACHE':'huggingface/hub',
        'HF_ASSETS_CACHE':'huggingface/assets','HF_XET_CACHE':'huggingface/xet',
        'TRANSFORMERS_CACHE':'huggingface/transformers','PIP_CACHE_DIR':'pip','UV_CACHE_DIR':'uv',
    }.items():
        env[name] = str(root/'cache'/relative)
    return env


def run_xxl(audio, output, engine, models, config, language, offset):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'processed.srt').is_file() and (output/'complete.json').is_file():
        return dict(exit_code=0, reused=True)
    number = 1
    attempt = output / f'attempt-{number:03d}'
    while attempt.exists():
        number += 1
        attempt = output / f'attempt-{number:03d}'
    command = [sys.executable, '-X', 'utf8', '-B', str(ROOT/'scripts/xxl_transcribe.py'),
               str(audio), '--prepared-audio', '--language', language, '--config', str(config),
               '--engine', str(engine), '--models', str(models),
               '--timeline-offset', str(offset), '--output-dir', str(attempt)]
    started = time.monotonic()
    environment = branch_environment(ROOT, output)
    environment.pop('GEMINI_API_KEY', None)
    with (output/'runner.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(command, env=environment, cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode == 0:
        for name in ('raw.srt','processed.srt','postprocess.json','run.json'):
            shutil.copy2(attempt/name, output/name)
        write_json(output/'complete.json', dict(attempt=attempt.name, complete=True))
    return dict(exit_code=result.returncode, elapsed_seconds=round(time.monotonic()-started, 2), attempt=attempt.name)


def run_gemini(audio, output, config, language, offset, dry_run=False):
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-X', 'utf8', '-B', str(ROOT/'scripts/gemini_pipeline.py'),
               str(audio), '--output-dir', str(output), '--config', str(config),
               '--language', language, '--timeline-offset', str(offset)]
    if dry_run:
        command.append('--dry-run')
    started = time.monotonic()
    with (output/'runner.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(command, env=branch_environment(ROOT, output), cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return dict(exit_code=result.returncode, elapsed_seconds=round(time.monotonic()-started, 2))


def xxl_paths(workflow):
    engine = Path(workflow['xxl_executable']).expanduser()
    models = Path(workflow['xxl_models']).expanduser()
    if not engine.is_absolute() or not engine.is_file() or not models.is_absolute() or not models.is_dir():
        raise ValueError('请在 workflow-config.json 中填写有效的 XXL 引擎和模型目录绝对路径')
    return engine.resolve(), models.resolve()


def main():
    parser = argparse.ArgumentParser(description='视频或音轨 → 公共音频 → 独立双路听写，供后续双字幕校订')
    parser.add_argument('media', type=Path, nargs='?')
    parser.add_argument('--language', default='')
    parser.add_argument('--audio-stream', type=int)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--job-dir', type=Path)
    parser.add_argument('--branch', choices=('both','gemini','xxl'), default='both')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list-tracks', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='仅准备音轨供现有双稿复听')
    parser.add_argument('--gemini-config', type=Path, default=ROOT/'gemini-config.json')
    args = parser.parse_args()
    workflow = workflow_config(ROOT)
    configure_ffmpeg(workflow)
    work_root = Path(workflow['work_root'])
    if args.job_dir:
        args.job_dir = task_output_path(args.job_dir, work_root)
    if args.check:
        engine, model_dir = xxl_paths(workflow)
        command = [sys.executable, '-X','utf8','-B', str(ROOT/'scripts/gemini_pipeline.py'),
               '--check', '--config', str(args.gemini_config.resolve())]
        if args.language:
            command.extend(['--language',args.language])
        checks = task_output_path(work_root/'checks', work_root)
        result = subprocess.run(command, env=branch_environment(ROOT, checks), check=False)
        if result.returncode:
            return result.returncode
        for name in ('east-asian.json','western.json'):
            cfg = json.loads((ROOT/'config'/name).read_text(encoding='utf-8-sig'))
            if cfg['vad_method'] != 'pyannote_v3':
                raise ValueError('XXL 配置应使用内置 pyannote_v3')
            if not (model_dir/('faster-whisper-'+cfg['model'])/'model.bin').is_file():
                raise ValueError('XXL 本地 Whisper 模型缺失')
        print('XXL 本地引擎、模型和参数配置检查通过')
        return 0
    if not args.media or not args.media.is_file():
        parser.error('需要可读取的输入音视频文件')
    media = args.media.resolve()
    probe = json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',str(media)]))
    tracks = [s for s in probe['streams'] if s['codec_type']=='audio']
    if args.list_tracks:
        print(json.dumps([dict(index=s['index'], codec=s['codec_name'], channels=s.get('channels'),
                              language=s.get('tags',{}).get('language'), title=s.get('tags',{}).get('title'))
                          for s in tracks], ensure_ascii=False, indent=2))
        return 0
    if not args.language and not args.prepare_only:
        parser.error('请用 --language 指定原声语言，如 ja 或 es')
    stream = args.audio_stream
    if stream is None and len(tracks)==1:
        stream = tracks[0]['index']
    if not any(s['index']==stream for s in tracks):
        raise ValueError('请先用 --list-tracks 查看音轨，再用 --audio-stream 指定全局索引')
    if not math.isfinite(args.start) or args.start<0 or (args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0)):
        raise ValueError('起点须非负，时长须为正数')
    config = ROOT/'config'/('east-asian.json' if args.language.split('-')[0] in ('ja','zh','ko') else 'western.json')
    selected = () if args.prepare_only else (('gemini','xxl') if args.branch=='both' else (args.branch,))
    if 'gemini' in selected:
        import gemini_transcribe as g
        gemini_config = g.read_config(args.gemini_config)
        if not (ROOT/'alignment-models.json').is_file():
            raise ValueError('请先使用 install_alignment_model.py 安装校时模型')
        models = json.loads((ROOT/'alignment-models.json').read_text(encoding='utf-8'))
        if args.language not in models:
            raise ValueError('当前已配置本地校时模型的语言为 '+', '.join(models))
        if not args.dry_run and not (os.environ.get('GEMINI_API_KEY') or gemini_config['api_key']).strip():
            raise ValueError('请填写 Gemini API Key')
    public_gemini = ({k:v for k,v in gemini_config.items() if 'key' not in k and k!='concurrency'}
                     if 'gemini' in selected else None)
    xxl_config = None
    if 'xxl' in selected:
        engine, model_dir = xxl_paths(workflow)
        xxl_config = json.loads(config.read_text(encoding='utf-8-sig'))
        xxl_config.update(engine_path=str(engine), model_dir=str(model_dir))
    stat = media.stat()
    identity = dict(media=str(media), size=stat.st_size, mtime_ns=stat.st_mtime_ns, stream=stream,
                    language=args.language, start=args.start, duration=args.duration,
                    gemini=public_gemini, xxl=xxl_config)
    if args.job_dir:
        job = args.job_dir
    else:
        if args.resume:
            raise ValueError('恢复任务时请指定 --job-dir')
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',media.stem).strip(' .')[:70] or 'media'
        job = task_output_path(
            work_root / f'{name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}', work_root
        )
    if args.resume:
        saved = json.loads((job/'input.json').read_text(encoding='utf-8'))
        if saved != identity:
            raise ValueError('恢复任务的原文件、音轨或参数与记录不同')
    else:
        job.mkdir(parents=True, exist_ok=False)
        write_json(job/'input.json',identity)
    with job_lock(job / 'run.lock'):
        shared = job/'shared'
        shared.mkdir(exist_ok=True)
        audio = shared/'audio.wav'
        preparation_path = shared/'preparation.json'
        if not audio.exists() or not preparation_path.exists():
            partial = shared/'audio.part.wav'
            partial.unlink(missing_ok=True)
            metadata = prepare_audio(media, partial, stream, args.start, args.duration)
            partial.replace(audio)
            write_json(preparation_path,metadata)
        print(f'任务目录 {job}', flush=True)
        if args.prepare_only:
            return 0
        status_path = job/'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        status['complete'] = False
        status['dry_run'] = args.dry_run
        write_json(status_path,status)
        if args.dry_run:
            status['gemini'] = run_gemini(audio,job/'gemini',args.gemini_config,args.language,args.start,True) if 'gemini' in selected else dict(skipped=True)
            status['xxl'] = dict(dry_run=True, config=str(config), shared_audio=str(audio))
            write_json(status_path,status)
            return status['gemini'].get('exit_code',0)
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            tasks = {}
            if 'gemini' in selected:
                tasks[executor.submit(run_gemini,audio,job/'gemini',args.gemini_config,args.language,args.start)]='gemini'
            if 'xxl' in selected:
                tasks[executor.submit(run_xxl,audio,job/'xxl',engine,model_dir,config,args.language,args.start)]='xxl'
            for future in as_completed(tasks):
                name = tasks[future]
                try:
                    status[name] = future.result()
                except (OSError,ValueError,RuntimeError) as error:
                    status[name] = dict(exit_code=1,error=type(error).__name__)
                print(f'{name} 退出码 {status[name]["exit_code"]}',flush=True)
                write_json(status_path,status)
        status['complete'] = all(status[name].get('exit_code')==0 for name in selected)
        write_json(status_path,status)
        if status['complete']:
            print('两路独立听写已完成，可按 reconcile-subtitles 进行双稿校订。' if len(selected)==2 else '听写分支已完成。',flush=True)
        return 0 if status['complete'] else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        message='配置或任务记录无效' if isinstance(error,json.JSONDecodeError) else str(error)
        print(message,file=sys.stderr)
        raise SystemExit(1)
