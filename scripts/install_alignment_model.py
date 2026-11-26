"""Download, validate and register an ONNX CTC model entirely under this tool."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from runtime_env import configure_runtime
from media_audio import job_lock
from model_registry import model_directory, validate_model

ROOT = Path(__file__).resolve().parents[1]
CACHES = configure_runtime(ROOT)


def main():
    parser = argparse.ArgumentParser(description='在 Gemini-Transcribe 内下载并登记新语种的 ONNX CTC 校时模型')
    parser.add_argument('--language')
    parser.add_argument('--repo-id')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--onnx-file', default='onnx/model_quantized.onnx')
    parser.add_argument('--method', choices=('character','word'))
    parser.add_argument('--model-name')
    parser.add_argument('--check-cache', action='store_true')
    args = parser.parse_args()
    if args.check_cache:
        print(json.dumps(dict(model_root=str(ROOT/'models'),cache_paths=CACHES,temporary=os.environ['TEMP']),ensure_ascii=False,indent=2))
        return 0
    if not args.language or not args.repo_id or not args.method:
        parser.error('需要 --language、--repo-id、--method')
    if not re.fullmatch(r'[a-z]{2,3}(?:-[a-z0-9]+)*',args.language):
        raise ValueError('语种代码无效')
    filename = PurePosixPath(args.onnx_file)
    if filename.is_absolute() or '..' in filename.parts or re.search(r'[\\:*?]',args.onnx_file):
        raise ValueError('onnx-file 必须是仓库内的相对文件名')
    name = args.model_name or (args.language+'--'+args.repo_id.replace('/','--'))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name):
        raise ValueError('模型目录名无效')
    entry = dict(directory='models/'+name,method=args.method)
    destination = model_directory(ROOT,entry)
    registry_path = ROOT/'alignment-models.json'
    os.environ['HF_HUB_OFFLINE']='0'
    os.environ['TRANSFORMERS_OFFLINE']='0'
    from huggingface_hub import HfApi, snapshot_download
    with job_lock(ROOT/'models-install.lock'):
        registry = json.loads(registry_path.read_text(encoding='utf-8-sig')) if registry_path.exists() else {}
        if args.language in registry:
            if registry[args.language]['directory'] != entry['directory']:
                raise ValueError('该语种已有模型。更换前请明确编辑 alignment-models.json 中对应语种的登记')
            print(json.dumps(validate_model(destination,args.method)))
            return 0
        if any(model_directory(ROOT, existing) == destination for existing in registry.values()):
            raise ValueError('该模型目录已由其他语种使用，请指定新的 model-name')
        if destination.exists() and any(destination.iterdir()) and not (destination/'download-source.json').exists():
            raise ValueError('该模型目录已有文件，请指定新的 model-name')
        pin = HfApi().model_info(args.repo_id,revision=args.revision).sha
        identity = dict(repo_id=args.repo_id,revision=pin,onnx_file=args.onnx_file,method=args.method)
        metadata_path = destination/'download-source.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('已有下载目录对应不同模型或版本，请使用新的 model-name')
        destination.mkdir(parents=True,exist_ok=True)
        metadata_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
        snapshot_download(repo_id=args.repo_id,revision=pin,local_dir=str(destination),
                          cache_dir=CACHES['HF_HUB_CACHE'],
                          allow_patterns=['config.json','vocab.json',args.onnx_file,args.onnx_file+'*',
                                          str(filename.parent/'*.onnx_data'),str(filename.parent/'*.data'),
                                          'preprocessor_config.json','tokenizer_config.json'])
        downloaded = destination/args.onnx_file
        if downloaded != destination/'model_int8.onnx':
            # The existing aligner uses this canonical name. A same-directory
            # rename preserves relative external-data references in ONNX files.
            if downloaded.parent != destination:
                import onnxruntime as ort
                # Validate that the supplied ONNX is self-contained before moving.
                ort.InferenceSession(str(downloaded),providers=['CPUExecutionProvider'])
            shutil.copy2(downloaded,destination/'model_int8.onnx')
            for data in downloaded.parent.glob('*.onnx_data'):
                if data.parent != destination:
                    shutil.copy2(data,destination/data.name)
            for data in downloaded.parent.glob('*.data'):
                if data.parent != destination:
                    shutil.copy2(data,destination/data.name)
        details = validate_model(destination,args.method)
        registry[args.language] = dict(entry,source=identity)
        temporary = registry_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(registry_path)
        print(json.dumps(dict(language=args.language,directory=str(destination),validation=details),ensure_ascii=False))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Hub exception messages may contain signed URLs. Keep those out of logs.
        if type(error).__module__.startswith(('huggingface_hub','requests','httpx')):
            print('模型下载失败，请检查仓库访问权限、文件名和网络连接',file=sys.stderr)
        else:
            print('模型安装失败：'+str(error),file=sys.stderr)
        raise SystemExit(1)
