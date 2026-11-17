# Reconcile Subtitles / 双字幕比对校订

Compare two timestamped subtitle files or speech recognition transcripts in the same language for the same recording. The skill aligns their timelines, selects a base draft, fixes supported errors, recovers missing lines, and flags unresolved differences.

比较同一段音视频的两份带时间码的同语种字幕或语音识别稿。对齐时间轴后选定底稿，校正有依据的错听、补回漏句，并标记证据不足的分歧。

## Install / 安装

Copy the entire `reconcile-subtitles` folder into a skills directory supported by your agent. Keep `SKILL.md` at the folder root. `agents/openai.yaml` provides optional UI metadata for Codex. The package includes no executable scripts and requires no bundled dependencies.

将整个 `reconcile-subtitles` 文件夹复制到你的智能体所支持的技能目录。将 `SKILL.md` 保留在技能根目录；`agents/openai.yaml` 是 Codex 的可选界面配置。包内没有可执行脚本，也没有附带依赖。

## Use / 使用

Provide both input files. A matching video or audio file is optional. The skill can check disputed lines against the recording when local media is accessible and the agent can listen to it. It does not upload media to an external transcription service without authorization.

提供两份输入字幕；如需复核原声，可以附上对应的音视频。环境无法复听时，技能仍可完成双稿校订并标记疑点。未经授权，不将音视频上传至外部转录服务。

> Use $reconcile-subtitles to compare `sample.a.srt` and `sample.b.srt`. Keep the dialogue in its original language and flag any uncertain lines.
>
> 用 $reconcile-subtitles 校订 `sample.a.srt` 和 `sample.b.srt`，保留对白的原语种，并标出无法确认的句子。

By default, you get a reconciled SRT and a CSV report of changes and open questions. The original files are kept intact.

默认交付校订后的 SRT 和记录改动与待核问题的 CSV 对照表，保留原始文件。
