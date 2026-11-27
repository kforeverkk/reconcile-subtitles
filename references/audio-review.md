# Hydrogen 待核语音复听

完成初次双稿比对后，使用同一剪辑的原声检查仍标为「待核」且能定位的片段。标准听写仍由 Gemini 3.8 Flash 和 XXL 分别完成。复听仅使用 `gemini-4-hydrogen`，沿用配置好的 Google Gemini API 项目及 Key，按本次任务授权发送音频。已有两份字幕并同时提供同一剪辑的原声时也可进入本步骤。

## 生成复听证据

复用配置的工作根目录下的本片任务目录。待核清单保存为 `reconciled/pending-ranges.json`，切片和返回结果保存到 `reconciled/hydrogen-review`。复听已有双稿时，只需 Python 3.12、FFmpeg、空白样例生成的 Gemini 配置及有效 Key。已有符合规格的共享音轨时可直接复用。

1. 复用 `shared/audio.wav` 和 `shared/preparation.json` 中的 `timeline_offset_ms`。原声单独提供时，从技能根目录用配置的 Python 运行 `scripts/transcribe_video.py <原声路径> --prepare-only`，按需加 `--audio-stream`。记录命令返回的本片任务绝对路径。
2. 建立 `pending-ranges.json`。每项填写唯一字符串 `id` 和原片坐标的整数 `start_ms`、`end_ms`，例如 `[{"id":"170","start_ms":1708388,"end_ms":1713410}]`。优先用实际存疑对白的时间，别把含长空档的整行对齐跨度直接当作疑点。若一行有几个分开的疑点，分别赋予可追溯到该行的 ID。清单只列定位信息。
3. 从技能根目录运行下面的命令，将 `$job` 设为本片任务的绝对路径。长范围会拆成多片，裁切时会保留少量前后文。Hydrogen 只收到音频和转写指令，候选字幕和既有判断留在本机。

```powershell
$job = Read-Host '本片任务的绝对路径'
$offset = (Get-Content -LiteralPath (Join-Path $job 'shared\preparation.json') -Raw | ConvertFrom-Json).timeline_offset_ms
$python = (Get-Content -LiteralPath ./workflow-config.json -Raw | ConvertFrom-Json).python_executable
& $python -X utf8 -B ./scripts/review_pending.py --audio (Join-Path $job 'shared\audio.wav') --ranges (Join-Path $job 'reconciled\pending-ranges.json') --out (Join-Path $job 'reconciled\hydrogen-review') --timeline-offset-ms $offset
```

`--dry-run` 可只核对清单和切片范围。复听证据保存在输出目录的 `review.json` 中，含原片坐标、原文和对应的 WAV。有效结果按切片内容与提示词缓存。请求部分失败时，退出码为 2，已有证据可继续使用。以相同参数重跑会复用成功结果并重试未完成的切片。

## 三稿校订

- 按 ID 对回初次比对记录，从 Hydrogen 的切片全文中定位对应对白。带时间戳的结果已换算成原片时间。只有文字的结果可支持读法，字幕时间沿用可核实的来源。相邻切片因保留前后文可能重复转写同一句。
- 结合两份初稿、Hydrogen 转写、前后文及片中证据逐项校订。Flash 与 Hydrogen 可能有相同误听。遇到人名、金额和否定等关键信息，检查其他证据后再定稿。明确的无人声返回只是一个模型判断。
- 对照表补记 Hydrogen 转写、请求状态、最终采用的读法和依据。仍有实质分歧、服务未返回可用证据或原声未覆盖的项目继续标「待核」。最后验收并交付 SRT 和 CSV。
