"""Shared audio preparation, reused from the tested XXL runner."""
import json
from pathlib import Path
import subprocess
from contextlib import contextmanager


def task_output_path(path, work_root):
    """Resolve a task output while rejecting paths outside the configured root."""
    if not work_root.is_absolute() or not path.is_absolute():
        raise ValueError('工作目录必须使用已配置任务根目录下的绝对路径')
    root = work_root.resolve()
    target = path.resolve()
    if root != work_root or target == root or not target.is_relative_to(root):
        raise ValueError(f'工作目录必须位于 {work_root} 的子目录中')
    return target


def workflow_config(root):
    path = root / 'workflow-config.json'
    if not path.is_file():
        raise ValueError('请根据 workflow-config.example.json 创建本地 workflow-config.json')
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    work_root = Path(config['work_root']).expanduser()
    if not work_root.is_absolute():
        raise ValueError('work_root 必须是绝对路径')
    work_root = work_root.resolve()
    if work_root.is_relative_to(root.resolve()):
        raise ValueError('work_root 必须位于技能目录之外，避免任务资料进入发布仓库')
    config['work_root'] = str(work_root)
    return config


@contextmanager
def job_lock(path):
    """Hold a Windows file lock so concurrent resumes cannot write one task."""
    import msvcrt
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('该任务已在运行，请等待当前进程完成') from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

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

