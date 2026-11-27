# 音视频输入的双路听写

输入为本地视频或音轨时，先完成两路独立听写，再继续 SKILL.md 中的双稿对齐、逐段校订和交付验收。两份字幕已有时直接使用双稿流程。

Gemini 的听写音频、原语种块整理文字和歌词分类文字发送至用户配置的接口，按用户的任务授权调用。密钥读取自本机工具目录的配置文件或对应环境变量。

## 工具和任务目录

- 先按 [首次使用与依赖检查](setup.md) 配置 Python、FFmpeg、XXL 和 Gemini。以下命令从技能根目录执行。接口配置为本地 `gemini-config.json`，主入口为 `scripts/transcribe_video.py`。
- XXL 引擎和模型路径来自本地 `workflow-config.json`。亚洲参数为 `config/east-asian.json`，欧美参数为 `config/western.json`。两组预设均使用 large-v2 与内置 `pyannote_v3`。
- 每次任务在 `workflow-config.json` 的 `work_root` 下生成本片唯一子目录。公共音轨放入 `shared/audio.wav`，选轨及原片起点记录在 `shared/preparation.json`。
- Gemini 各阶段字幕在 `gemini` 中，最终听写稿为 `gemini\processed.srt`。请求切片在 `gemini\work\chunks`，窗口恢复记录为 `gemini\work\window-*.json`，校时变化在 `gemini\timing.json`。
- XXL 最终听写稿为 `xxl\processed.srt`，原始 SRT/JSON 和日志位于 `xxl\attempt-*\engine` 及对应日志文件。最终校订字幕和 CSV 放入同一任务的 `reconciled` 子目录。

FFmpeg 和 ffprobe 优先从配置的 `ffmpeg_bin` 调用，也可由 PATH 或技能目录下已有的 `bin/` 提供。公共音轨只抽取一次，两路复用相同的 16 kHz 单声道 16 位 PCM WAV。抽轨保留时间戳空档，局部片段记录原片起点。两路各自断句、校时，输出时间均对应原片坐标。

## 调用和恢复

使用 PowerShell 7。先检查工具、查看音轨并确定原声语种。单音轨自动选择，多音轨且用户未指定时请用户选择容器全局索引。

```powershell
$python = (Get-Content -LiteralPath ./workflow-config.json -Raw | ConvertFrom-Json).python_executable
& $python -X utf8 -B ./scripts/transcribe_video.py --check --language ko
& $python -X utf8 -B ./scripts/transcribe_video.py 'sample.mkv' --list-tracks
& $python -X utf8 -B ./scripts/transcribe_video.py 'sample.mkv' --language ko --audio-stream 1
```

默认同时运行两路。`--start` 和 `--duration` 用于用户指定的原片区间。`--dry-run` 验证抽轨和 Silero 切分。单路任务可用 `--branch gemini` 或 `--branch xxl`。主入口会显示任务目录，也可用 `--job-dir` 指定工作根目录下的绝对路径。后续校订时沿用该路径，成稿直接写到其 `reconciled` 子目录。

任务结束后检查 `status.json`、两路退出码、Gemini 的 report.json 和两份 processed.srt。失败时查看相应 runner.log，修复问题后用相同输入、参数及 `--job-dir` 加 `--resume` 恢复。完成的窗口和阶段会复用，XXL 失败尝试保留在独立 attempt 目录中。

## 每路处理顺序

Gemini 使用 Silero 静音切点生成约180秒窗口，默认MP3、并发5。空响应、异常时间码、截断及疑似生成陷环使用已有的重试、格式切换或拆短逻辑。服务端明确的无人声判断单独记录。随后依次进行原语种块整理、本地 CTC 校时、歌词分类及最终排版，保留 raw、refined、aligned 和 processed 各阶段字幕。供双稿比对使用的是本地校时后的 processed 稿。

XXL 使用对应语种参数组进行听写和词级定时，再执行已验证的词级与波形起点微调、可靠长句拆分、短残句合并、补短显示时长和字幕内部换行。保留来源编号、边界变化及警告。

两路各自完成后，读取两份 processed 字幕的完整内容，按 SKILL.md 校订实质分歧和未匹配段。使用共享音轨对仍待核且能定位的片段进行 [Hydrogen 复听](audio-review.md)，再验收并交付最终 SRT 和 CSV。

## 新语种的本地校时模型

首次使用音视频模式时，按原声语种安装兼容当前 character 或 word CTC 校时器的 ONNX 模型。工具需要模型权重、`config.json` 和 `vocab.json`。安装后会生成本地 `alignment-models.json`。下例为韩语模型，其他语种替换模型仓库、版本、文件名和校时方式。模型的访问权限和许可见其上游仓库。

```powershell
& $python -X utf8 -B ./scripts/install_alignment_model.py --language ko --repo-id 'XDcobra/wav2vec2-large-xlsr-53-korean-ONNX' --revision 'a6162331bac7e12c609ffe34c0a641b3efa6df62' --onnx-file 'onnx/model_int8.onnx' --method word
```

新模型全部保存到工具根目录的 models 子目录，来源与固定版本写入 download-source.json。安装工具显式指定本地目录及下载缓存，验证 CTC 配置、字表和 ONNX 输出后才登记语种。登记路径必须属于本工具的 models 目录。

Hugging Face 下载缓存、Torch 缓存和默认临时文件由入口定向到技能根目录下的 `cache/`。任务阶段的临时文件使用该任务下的临时目录。可用安装脚本的 `--check-cache` 查看缓存路径。
