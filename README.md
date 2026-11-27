# Reconcile Subtitles · 双字幕比对校订

在 Codex 中校订同一段音视频的两份同语种字幕。技能会对齐时间轴、纠正有依据的错听、补回漏句，并把无法确认的对白标为「待核」。Windows 上也可从本地音视频开始，先生成两份独立听写稿，再校订。

A Codex skill that reconciles two same-language subtitle drafts and flags unresolved lines. On Windows, it can also generate two independent drafts from local media and review disputed clips. [English quick start](#english-quick-start)

## 安装并开始

1. 下载或克隆本仓库，将 `reconcile-subtitles` 文件夹放入 Codex 的技能目录，例如 `~/.codex/skills/`。确认 `SKILL.md` 位于该文件夹根目录。
2. 在 Codex 中调用 `$reconcile-subtitles`，并提供文件路径。可以直接复制下面的示例，换成自己的文件名。

| 你有什么 | 在 Codex 中这样说 | 需要什么 |
| --- | --- | --- |
| 两份现成字幕 | 用 `$reconcile-subtitles` 校订 `draft-a.srt` 和 `draft-b.srt`。 | 可写的工作目录 |
| 两份字幕和同一剪辑的原声 | 用 `$reconcile-subtitles` 校订这两份字幕，再用 `sample.mkv` 复听待核片段。 | Windows、PowerShell 7、Python、Gemini API。需要抽轨时还要有 FFmpeg 和 ffprobe |
| 一份音频或视频 | 用 `$reconcile-subtitles` 听写 `sample.mkv`，比对双稿并复听待核片段。 | Windows、PowerShell 7、Python 及其依赖、FFmpeg 和 ffprobe、Whisper XXL 及其模型、对应语种的 ONNX 校时模型、Gemini API |

## 第一次使用会发生什么

1. **选择工作目录。** Codex 会请你提供一个位于技能目录之外的可写绝对路径，并保存到本地 `workflow-config.json`。每部影片会在该位置下使用独立任务目录。
2. **检查依赖。** Windows 上会检查已配置或可发现的 Python、FFmpeg、Whisper XXL、相关模型和 Gemini 配置，并说明哪些缺项会影响当前任务。只比对两份字幕时，确认工作目录后即可开始。
3. **决定谁来安装。** 如果需要 Codex 帮忙，它会先询问具体安装位置，安装完成后复查。如果选择自行安装，Codex 会等你告知安装完成，再询问实际路径并复查。已验证的工具路径会存到本地配置，供下次使用。

也可以按 [首次使用与依赖检查](references/setup.md) 手动配置。听写依赖和命令见 [双路听写](references/transcription.md)，已有双稿的原声复听见 [待核复听](references/audio-review.md)。

## 听写和复听用什么模型

| 阶段 | 模型 | 处理内容 |
| --- | --- | --- |
| Gemini 首轮听写 | 默认 `gemini-3.8-flash` | 将所选音轨分段发送到配置的 Gemini 接口。首轮模型可通过 `model` 配置 |
| 本地首轮听写 | Purfview Faster Whisper XXL | 在本机独立听写同一音轨。随包预设使用 large-v2 和内置 pyannote_v3 |
| Gemini 稿整理与校时 | 文本默认沿用首轮模型，时间轴由本地 ONNX CTC 校准 | 原语种整理与歌词识别会发送字幕文字。文本模型可用 `text_model` 单独配置 |
| 双稿比对后的待核复听 | `gemini-4-hydrogen` | 仅向 Google Gemini 发送待核原声片段和转写指令 |

使用 Google Gemini 官方接口时，首轮听写和复听可共用同一个 API 项目和 Key。调用前按当前任务的授权确认发送范围。Key 可通过环境变量 `GEMINI_API_KEY` 提供，或填入从样例复制的本地 `gemini-config.json`。真实配置由 Git 忽略。第三方工具和模型由使用者安装。

## 输出文件

在所选工作目录的本片任务下，打开 `reconciled/`：

- `<片名>.reconciled.srt` 保存校订后的原语种字幕。
- `<片名>.comparison.csv` 记录两份原稿、采用的读法、校订依据及仍待核的条目。

## English quick start

1. Copy the `reconcile-subtitles` folder into your Codex skills directory and invoke `$reconcile-subtitles` with two subtitle files, two subtitles plus matching media, or one local media file.
2. On first use, choose a writable work directory outside the skill folder. On Windows, Codex checks dependencies. If you want help installing missing tools, Codex asks where to install them first. If you prefer to install them yourself, tell Codex when you are done, then provide their paths for verification.
3. For subtitle-only tasks, the work directory is sufficient. Full media transcription uses Gemini 3.8 Flash by default and local Whisper XXL. Gemini 4 Hydrogen reviews only disputed audio clips after the drafts have been compared. The final SRT and CSV are saved in the task’s `reconciled/` directory.

See the [first-run setup guide](references/setup.md), [transcription guide](references/transcription.md), and [audio review guide](references/audio-review.md).
