# Reconcile Subtitles / 双字幕比对校订

A Codex skill for reconciling two same-language subtitle drafts. On Windows, it can also generate independent drafts with Gemini Flash and Purfview Faster Whisper XXL, then use Gemini Hydrogen for targeted audio review. It delivers a reconciled SRT and a CSV comparison with unresolved readings marked.

这是一个用于比对并校订同一段音视频的两份同语种字幕的 Codex 技能。Windows 上也可从音视频开始，先用 Gemini Flash 与 Purfview Faster Whisper XXL 分别听写，再用 Gemini Hydrogen 复听待核片段。交付 SRT 和 CSV 对照表，保留证据不足的分歧。

## 安装 / Install

将整个 `reconcile-subtitles` 目录放进 Codex 的技能目录，例如 `~/.codex/skills/`。保持 `SKILL.md` 在该目录根部，然后在 Codex 中调用技能。

Copy the entire folder into your Codex skills directory and invoke the skill.

首次使用时，Codex 会请你选择技能目录之外的工作路径，并在 Windows 上检查依赖。缺项会列出来，你可以选择让 Codex 协助安装，或自行安装后提供路径。路径保存在本地 `workflow-config.json`。**只比对两份现成字幕时，有可写的工作目录即可开始。**

On first use, Codex asks where to store task files and checks the Windows dependencies. You can ask Codex to install missing tools in locations you choose, or install them yourself and provide their paths. Subtitle-only tasks require only the work directory. See [first-run setup](references/setup.md).

> 用 $reconcile-subtitles 校订 `draft-a.srt` 和 `draft-b.srt`，保留原语种，并标记仍需核对的对白。

## 音视频模式 / Audio and video mode

使用上述引导，也可以手动从 `workflow-config.example.json` 创建配置。在其中填写工作目录、Python 可执行文件、FFmpeg 目录、Purfview Faster Whisper XXL 可执行文件和 Whisper 模型父目录。随包预设使用 CUDA、float16、large-v2 和内置 pyannote_v3，可在 `config/` 中调整设备和模型。用配置的 Python 安装依赖；以下以自选目录中的虚拟环境为例：

```powershell
$venv = Read-Host '虚拟环境的安装绝对路径'
py -3.12 -m venv $venv
& (Join-Path $venv 'Scripts/python.exe') -m pip install -r requirements.txt
Copy-Item gemini-config.example.json gemini-config.json
```

在当前环境中提供 `GEMINI_API_KEY`，或把 Key 填入已被 Git 忽略的本地 `gemini-config.json`。按[听写配置](references/transcription.md)为影片语种登记一个兼容的本地 ONNX CTC 校时模型，再用 `--check` 检查依赖和模型。

Install Python 3.12, FFmpeg, ffprobe, and XXL with a compatible Whisper model. Set the XXL paths, install the Python dependencies, configure your Gemini credential, and register a language-appropriate ONNX CTC alignment model. See the [transcription reference](references/transcription.md) and the [review reference](references/audio-review.md) for commands.

> 用 $reconcile-subtitles 处理 `sample.mkv`，按原语种独立听写两份字幕，并对双稿比对后的待核片段进行复听。

### Gemini 模型与发送内容 / Models and data sent

- **首轮听写**默认使用 `gemini-3.8-flash`，按分段发送所选音轨。首轮模型可在 `gemini-config.json` 的 `model` 字段中配置。
- **待核复听**在双稿比对后使用 `gemini-4-hydrogen`，仅发送能定位到原声的待核片段。两阶段可复用同一个 Google Gemini API 项目和 Key。
- **原语种整理与歌词识别**发送字幕文字。默认沿用首轮模型，也可以通过 `text_model` 单独配置。调用前按本次任务授权确认传输范围。

The first Gemini transcription pass defaults to `gemini-3.8-flash`. Targeted audio review uses `gemini-4-hydrogen` only for unresolved clips after comparing the two drafts. Text cleanup and lyric detection send subtitle text and default to the first-pass model.

Python 依赖、第三方引擎和模型各自依据其上游条款提供。发布包包含技能说明、调用脚本和空配置样例。第三方二进制、模型权重、真实凭据和示例影片均由使用者自行准备。
