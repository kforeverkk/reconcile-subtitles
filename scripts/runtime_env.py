"""Keep application-controlled downloads, caches and temporary files together."""
import os
from pathlib import Path
import tempfile


def configure_runtime(root, temporary=None):
    root = Path(root).resolve()
    cache = root / 'cache'
    paths = {
        'HF_HOME': cache / 'huggingface',
        'HF_HUB_CACHE': cache / 'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': cache / 'huggingface/hub',
        'HF_ASSETS_CACHE': cache / 'huggingface/assets',
        'HF_XET_CACHE': cache / 'huggingface/xet',
        'TRANSFORMERS_CACHE': cache / 'huggingface/transformers',
        'TORCH_HOME': cache / 'torch',
        'XDG_CACHE_HOME': cache,
        'PIP_CACHE_DIR': cache / 'pip',
        'UV_CACHE_DIR': cache / 'uv',
        'CUDA_CACHE_PATH': cache / 'cuda',
    }
    for name, path in paths.items():
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('缓存目录实际指向工具目录外部')
        paths[name] = path
        os.environ[name] = str(path)
    temporary = Path(temporary) if temporary else cache / 'temp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = str(temporary)
    os.environ['TMP'] = str(temporary)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    tempfile.tempdir = str(temporary)
    return {name:str(path) for name,path in paths.items()}
