# Reconcile Subtitles / 双字幕比对校订

A Codex skill for reconciling two same-language subtitle drafts. On Windows, it can also generate independent drafts with Gemini Flash and Purfview Faster Whisper XXL, then use Gemini Hydrogen for targeted audio review. It delivers a reconciled SRT and a CSV comparison with unresolved readings marked.

这是一个用于比对并校订同一段音视频的两份同语种字幕的 Codex 技能。Windows 上也可从音视频开始，先用 Gemini Flash 与 Purfview Faster Whisper XXL 分别听写，再用 Gemini Hydrogen 复听待核片段。交付 SRT 和 CSV 对照表，保留证据不足的分歧。

## 安装 / Install

将整个 `reconcile-subtitles` 目录放进 Codex 的技能目录，例如 `~/.codex/skills/`。保持 `SKILL.md` 在该目录根部。以下命令在技能目录下运行。

Copy the entire folder into your Codex skills directory. Run the setup commands from the installed skill directory.

```powershell
Copy-Item workflow-config.example.json workflow-config.json
```

在本地 `workflow-config.json` 中把 `work_root` 填成技能目录之外的可写绝对路径。Windows JSON 路径可用正斜杠。任务中间文件和最终输出都保存在该根目录的本片子目录。**只比对两份现成字幕时，设置这一个字段即可，无需 Python、音频工具或 API Key。**

Set `work_root` to a writable absolute path. Comparing two existing subtitle files requires only this field. Each task writes its work and deliverables into its own directory under this root.

> 用 $reconcile-subtitles 校订 `draft-a.srt` 和 `draft-b.srt`，保留原语种，并标记仍需核对的对白。

## 音视频模式 / Audio and video mode

1. 在 Windows 安装 Python 3.12，并让 [FFmpeg 和 ffprobe](https://ffmpeg.org/download.html) 可从 PATH 调用。下载安装 [Purfview Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win) 及所选模型。随包预设使用 CUDA、float16、large-v2 和内置 pyannote_v3。用户可编辑 `config/` 中的设备和模型参数以适配本机。
2. 在 `workflow-config.json` 填写 `xxl_executable` 和 `xxl_models` 的绝对路径。后者指向包含 `faster-whisper-large-v2/model.bin` 的模型父目录。
3. 准备本地 Python 环境和 Gemini 配置。

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item gemini-config.example.json gemini-config.json
```

在当前环境中提供 `GEMINI_API_KEY`，或把 Key 填入已被 Git 忽略的本地 `gemini-config.json`。按[听写配置](references/transcription.md)为影片语种登记一个兼容的本地 ONNX CTC 校时模型，再用 `--check` 检查依赖和模型。

Install Python 3.12, FFmpeg, ffprobe, and XXL with a compatible Whisper model. Set the XXL paths, install the Python dependencies, configure your Gemini credential, and register a language-appropriate ONNX CTC alignment model. See the [transcription reference](references/transcription.md) and the [review reference](references/audio-review.md) for commands.

> 用 $reconcile-subtitles 处理 `sample.mkv`，按原语种独立听写两份字幕，并对双稿比对后的待核片段进行复听。

Gemini 听写和复听会把所选音频发送到 Google Gemini。原语种整理与歌词识别会发送字幕文字。调用前按本次任务授权确认传输范围。复听复用同一个 API 项目和 Key，默认模型为 `gemini-4-hydrogen`。

Python 依赖、第三方引擎和模型各自依据其上游条款提供。发布包包含技能说明、调用脚本和空配置样例。第三方二进制、模型权重、真实凭据和示例影片均由使用者自行准备。
