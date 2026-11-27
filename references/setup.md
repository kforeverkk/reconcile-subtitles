# 首次使用与依赖检查

这是 Codex 在调用技能时执行的流程。Windows 音频模式使用 PowerShell 7。可先做只读检查，再集中向用户报告缺项。

## 确认工作目录

1. 第一次使用时，询问用于保存所有任务文件和成品的工作根目录。用户已明确给出时可直接采用。要求使用技能目录之外的可写绝对路径。没有答案时等待。
2. 将选定路径保存到技能根目录的本地 `workflow-config.json` 的 `work_root`。文件缺失时从 `workflow-config.example.json` 创建。只更新本次核实的字段，保留已有设置。确认可以建立任务子目录后再处理素材。

## 自动发现依赖

从技能根目录运行以下命令。该检查能在没有 Python 的情况下运行，输出所检查项目的状态和发现的路径。

```powershell
pwsh -NoProfile -File ./scripts/check_setup.ps1 -Mode media
```

- Windows 首次使用先检查音视频模式的工具。知道原声语种时加 `-Language` 检查相应校时模型。后续按实际任务复查。只有两份字幕时用 `-Mode subtitles` 检查工作目录，音频缺项可延期。其它平台的字幕模式可直接核对 `work_root`。
- 使用两份字幕和原声做复听时用 `-Mode review`。已有合规共享 WAV 时加 `-PreparedAudio`。此模式只检查 Python、Gemini 配置，以及需要抽轨时的 FFmpeg。完整视频听写还检查 Python 依赖、XXL 和 Whisper 模型。
- 优先使用本地配置的路径。未配置时查已有虚拟环境、Python Launcher 和 PATH。找到可用安装时，保存经核实的 `python_executable`、`ffmpeg_bin`、`xxl_executable` 和 `xxl_models`。若用户机器上已有未发现的安装，请让用户提供路径。
- `gemini_config` 表示本机配置和密钥是否已提供。提示用户按 README 本地配置，输出中只报告是否存在。模型登记缺失时按[本地校时模型说明](transcription.md#新语种的本地校时模型)处理。

## 与用户确认安装

1. 列出缺失或不可用的依赖及其对应功能，问用户是否需要 Codex 帮助安装。得到答复前等待。
2. **选择 Codex 安装**时，下一条询问每项的具体安装目录。Python 本体和虚拟环境可使用不同位置。Python 包安装到用户确认的解释器环境。下载模型时明确告知实际写入的位置。CTC 校时器使用技能的 `models/`，安装前确认其具体子目录。收到路径并具备所需权限后，使用官方安装包或项目发布页，在选定位置安装。
3. **选择自行安装**时，等用户确认安装完成，再询问这些项目的安装位置。问清 Python 可执行文件或虚拟环境、含 `ffmpeg.exe` 与 `ffprobe.exe` 的目录、XXL 可执行文件和模型父目录。相关音频步骤在缺项补齐前等待，字幕模式可继续。
4. 保存确认过的路径后重新运行检查。完整听写再用配置的解释器运行 `scripts/transcribe_video.py --check --language <语种>`。只有当前入口所需检查通过后才启动音频任务。

官方来源见 [Python for Windows](https://www.python.org/downloads/windows/)、[FFmpeg](https://ffmpeg.org/download.html) 和 [Purfview Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win/releases)。Python 3.12 是推荐版本。在确定好的环境里运行 `<python> -m pip install -r requirements.txt`。安装过程中使用的凭据和路径只存于本机配置。
