# 本机 LLM 启动记录

> 更新：`turboquant-hip-dflash2` 分支已接入 turbo4 KV，并使用本模型和 DFlash2 运行到有效上下文 `262144`，完成 42108 token 输入及实际生成。当前启动方式与实测结果见 [TurboQuant HIP 记录](turboquant-hip.md)。下文保留此前 main 分支的 q8_0/q4_0 配置作为基线。

本文记录 RX 7900 XTX / gfx1100、HIP 后端下的启动配置和历史经验。除非另有明确说明，本文记录的服务启动命令统一显式使用 `--reasoning-effort medium`；上下文长度均包含输入与输出。能启动、能处理普通请求，不代表已经通过接近最大长度的满上下文压力测试。

## 1. 模型与上下文记录

| 主模型 | K / V 缓存 | 上下文 | 记录状态 |
|---|---|---|---|
| `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf` | `q8_0 / q4_0` | `262144` (256 Ki tokens) | 此前运行配置，健康检查正常，已正常生成；未验证满 256K 请求 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `turbo4_0 / turbo4_0` | `262144` (256 Ki tokens) | 当前运行配置，`-ub 512`；已完成约 42108 token 输入和真实生成测试，用户确认正常；未验证满 256K |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `q8_0 / turbo4_0` | `200000` (actual `200192`) | 当前运行配置；已通过健康检查和最小生成；`210000` 在约 190K 输入的实际 decode 阶段 OOM；在 `65536` 完成贪吃蛇测速，55.51 tok/s |
| 当时使用的模型，文件名未单独确认 | `q4_0 / q4_0` | 用户描述约 250K | 用户实测反馈；未记录精确 `-c` 数值，不等同于已验证 `262144` |
| `Qwen3.8-27B-heretic-ara.ROCmFP4-STRIX.gguf` | `q4_0 / q4_0` | `262144` | `scripts/serve-dflash2.sh` 的默认参考配置及既有文档经验，不是当前模型 |

当前草稿模型：

```text
/data/project/ai/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

不同量化文件的权重占用不同，不能只根据同为 27B 就套用相同上下文上限。比较时还应固定草稿模型、视觉编码器、物理批大小和其他显存使用情况。

## 2. 当前实际启动命令

在项目根目录执行。以下按运行进程和环境记录，不额外添加尚未应用的建议参数：

```bash
export LLAMA_ATTN_ROT_DISABLE=1

./build-rocmfp4/bin/llama-server \
  -m /data/project/ai/models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --model-draft /data/project/ai/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type ngram-map-k4v,draft-dflash \
  --spec-draft-n-max 5 \
  --spec-draft-p-min 0.4 \
  --spec-ngram-map-k4v-size-n 12 \
  --spec-ngram-map-k4v-size-m 48 \
  --spec-ngram-map-k4v-min-hits 1 \
  -ngld 99 \
  -ctk turbo4_0 -ctv turbo4_0 \
  -ngl 99 -c 262144 \
  -b 2048 -ub 512 \
  -fa on --jinja \
  --reasoning-effort medium \
  --no-kv-unified --parallel 1 \
  --ctx-checkpoints 2 --cache-ram 4096 \
  --no-warmup --no-mmproj \
  --host 0.0.0.0 --port 1234
```

本次 turbo4 中等思考启动日志：`/tmp/llama-turbo4-262144.log`。该路径位于临时目录，不保证重启后保留。

当前显式设置 `--reasoning-effort medium`，没有显式设置 `HSA_ENABLE_SDMA`、`--fit off`、外部聊天模板、思考开关、输出解析格式或思考预算。`--no-mmproj` 禁止自动加载视觉投影文件，当前也没有通过 `--mmproj` 指定文件，因此本次服务未启用视觉输入。

### 当前采样默认值

以下来自运行服务的 `/props`，API 请求可以覆盖这些默认值：

| 参数 | 当前值 |
|---|---:|
| temperature | 1.0 |
| top_k | 20 |
| top_p | 0.95 |
| min_p | 0.05 |
| repeat_penalty | 1.0 |
| presence_penalty | 0.0 |

## 3. 已有启动经验

### HIP 与量化 KV

本机混合架构的量化 KV 配置保留：

```bash
export LLAMA_ATTN_ROT_DISABLE=1
```

项目既有运行记录指出，未禁用上游注意力旋转时可能触发当前混合架构 HIP 路径的问题。不要为了量化精度直接删除此变量。纯 F16 KV 不需要依靠它规避量化 KV 旋转问题。

### DFlash2 与 ngram

保留以下已有组合：

```bash
--spec-type ngram-map-k4v,draft-dflash \
--spec-draft-n-max 5 --spec-draft-p-min 0.4 \
--spec-ngram-map-k4v-size-n 12 \
--spec-ngram-map-k4v-size-m 48 \
--spec-ngram-map-k4v-min-hits 1
```

ngram 对重复内容收益明显，DFlash2 收益取决于草稿接受率。不能用不同内容的 tok/s 判断某个启动参数是否更快。既有文档中的吞吐数字属于原测试条件，不保证当前模型复现。

启动时若在内存试测阶段出现：

```text
dflash requires ctx_other to be set (this warning is normal during memory fitting)
```

不应仅凭此信息认定启动失败。当前服务出现过此警告，随后正常加载并提供服务。`--fit off` 可以跳过自动适配，但不会减少真实运行所需显存；配置放不下时仍会失败。

### 混合架构与历史缓存

当前保留：

```bash
--no-kv-unified --parallel 1
```

既有项目经验是 DeltaNet 递归状态与历史复用需要特别处理；以上配置用于当前单槽位服务。不要未经测试改为多槽位或统一 KV。

需要纠正旧文档中的简化表述：`--ctx-checkpoints` 不是每轮 DFlash2 推测回滚快照的数量，也不能直接按数量估算显存占用。

- `tools/server/server-context.cpp` 的 `create_checkpoint()` 管理历史上下文检查点。
- 每轮推测解码另有 `slot.spec_ckpt`。
- 检查点的序列化数据保存在 `common_prompt_checkpoint` 的主机侧 `std::vector<uint8_t>` 中，见 `common/common.h`。
- 更多历史检查点可能减少编辑旧消息、切换历史分支时的重算，但会增加系统内存和复制开销。
- 当前 `--ctx-checkpoints 2`；未显式设置的 `--checkpoint-min-step` 默认是 `8192`。
- `--cache-ram 4096` 是主机 prompt 缓存容量上限，单位 MiB；不是 GPU KV 容量，不应理解为将当前活动 KV 自动换出到内存。

示例中的 `--ctx-checkpoints 64 --checkpoint-min-step 2048 --cache-ram 16384` 偏向保留更多历史恢复点，不是通用提速配置，也不会直接提高草稿接受率。

### 物理批大小与视觉编码器

- 当前 `-b 2048 -ub 512`。turbo4 使用 `turbo4_0/turbo4_0`、`-c 262144`，已完成约 42108 token 输入和真实生成测试，未出现 OOM；详细结果见 [TurboQuant HIP 记录](turboquant-hip.md)。此前 q8_0/q4_0 配置在 `-ub 512` 首次请求 OOM，不能套用于当前 turbo4 配置。
- 混合 KV `-ctk q8_0 -ctv turbo4_0` 的探测结果需要区分启动和实际长输入：当前使用 `-c 200000`，`/props` 实际报告 `n_ctx = 200192`，已通过健康检查和最小生成。`-c 210000` 实际报告 `n_ctx = 210176`，但约 189981 个输入 token 在 decode 阶段触发 `ROCm error: out of memory`，不能视为可用的长上下文上限。`-c 229376` 在当前二进制加载 DFlash2/MTP 时也因显存不足失败。该探测不是满上下文压力测试，当前 200K 配置仅确认可启动和处理短请求。
- `--no-warmup` 会让部分运行时分配延迟到真实请求；启动成功不等于推理可用。降低 `-ub` 可以降低部分计算缓冲和临时显存压力，但不会缩小 KV，prefill 也可能变慢。
- `--no-mmproj-offload` 用 CPU 运行视觉编码器，用于需要图片输入又想减少 GPU 占用的情况，图片处理通常更慢。
- `HSA_ENABLE_SDMA=0` 是 ROCm 传输路径的兼容性选项，不应未经对比就作为必需的加速参数。
- `--alias qwen` 只提供稳定的 API 模型别名，不影响推理性能。

### q8 K + turbo4 V 混合配置的降速与修复

日常配置继续使用 `-ctk turbo4_0 -ctv turbo4_0`。混合配置服务必须监听 `0.0.0.0`，不能使用 `127.0.0.1`，否则其他机器无法访问 1234 端口。后续启动混合 KV 服务统一使用 `--host 0.0.0.0`。修复前，`q8_0 K + turbo4_0 V` 虽能加载并完成极短请求，但贪吃蛇生成测试出现断崖式降速，不应将此前的启动探测当作稳定可用结论。

本次使用 `build-turbo4/bin/llama-server`、相同主模型及 DFlash2、`-c 65536 -b 2048 -ub 512`，请求设置 `max_tokens: 65535`、`temperature: 0` 并关闭思考。混合 KV 在生成 125 token 时累计速度为 41.17 tok/s，530 token 时降至 21.43 tok/s，1026 token 时仅 16.40 tok/s，1891 token 时仅 11.49 tok/s；测试中止，未取得完整生成结果。此前相同贪吃蛇请求的双 turbo4 完整生成速度为 54.99 tok/s，但输出内容和长度不同，不是严格同 token 性能对照。

对 `-c 210000` 的实际长输入复测：约 189981 个输入 token 在 decode 阶段触发 ROCm OOM，服务端断开请求；短请求不能证明接近 `n_ctx` 的长输入可用。

修复前，`ggml/src/ggml-cuda/fattn.cu` 要求 K/V 同时为 turbo4 才能选择 turbo4 GPU Flash Attention 向量内核，混合配置返回 `BEST_FATTN_KERNEL_NONE`。现已仅为 `q8_0 K + turbo4_0 V` 补齐 128/256 维内核分发、模板实例及 HIP/CUDA 构建列表，复用现有 q8 K 点积和 turbo4 V 解码，不扩展其他混合组合。

已重新编译 `build-turbo4`，GPU 数值测试覆盖 128/256 维、1/5 个 query，双 turbo4 与混合组合共 8 项全部通过，注意力最大绝对误差均小于 0.002。相同贪吃蛇请求下，修复后的混合 KV 完整生成 11981 token，服务端速度 55.51 tok/s，墙钟速度 55.43 tok/s，总耗时 216.16 秒；本次未再出现此前的断崖式降速。此前双 q8 为 56.04 tok/s，双 turbo4 为 54.99 tok/s；不同输出长度下的单次生成结果不能视为严格同 token 对照，也没有证明混合 KV 的质量收益或长上下文稳定性。

结果：`results/snake-q8k-turbo4v-patched.json`。修复前日志：`/tmp/kv-snake-q8k-turbo4v.log`；修复后日志：`/tmp/kv-snake-q8k-turbo4v-patched.log`；编译日志：`/tmp/build-mixed-turbo4.log`。临时目录中的文件不保证重启后保留。

## 4. 正确设置思考强度

当前主模型内置聊天模板已通过运行服务 `/props` 确认支持 `reasoning_effort`，无需另找 `chat_template.jinja`。

| 强度 | 当前模板行为 |
|---|---|
| `xhigh` | 默认值，额外注入仔细思考、验证假设和考虑替代方案的系统指令 |
| `medium` | 不注入额外的强度指令，仍保持思考模式 |
| `low` | 注入简短、聚焦思考的系统指令 |

当前 UD 模型模板接受 `xhigh`、`medium`、`low`，并将 `high` 映射为 `xhigh`；此前 ROCmFP4 STRIX 模型模板不包含该别名。不要套用其他模型的 `minimal`、`max`。当前启动显式使用 `--reasoning-effort medium`；后续双 turbo4 和混合 KV 启动都必须保留该参数；单次请求可以覆盖。

推荐通过当前版本的专用参数设置：

```bash
--reasoning on --reasoning-effort medium --reasoning-format deepseek
```

以下写法也有效，设置强度时与 `--reasoning-effort medium` 二选一：

```bash
--chat-template-kwargs '{"reasoning_effort":"medium"}'
```

注意三者分工：`--reasoning` 控制思考开关；`--reasoning-effort` 是模板提示层面的软控制；`--reasoning-format deepseek` 将思考解析到 `reasoning_content`，本身不提高强度。

如果需要明确限制思考长度，可以另外设置 `--reasoning-budget 4096`。默认 `-1` 不限制，`0` 触发立即结束思考，正数设置 token 预算。预算耗尽会触发结束思考机制，复杂任务可能受影响，不建议把 4096 当作所有任务的必需值。

OpenAI 兼容聊天请求也可逐次设置，无需重启：

```json
{
  "model": "/data/project/ai/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
  "messages": [{"role": "user", "content": "你的问题"}],
  "reasoning_effort": "medium",
  "reasoning_budget_tokens": 4096
}
```

## 5. KV 容量与 TurboQuant 的结论

此前 main 分支未移植 TurboQuant；新增分支的 turbo4 实测见本文顶部链接。以下是块存储成本，不能直接当作本机上下文上限：

| 类型 | 实际 bit/值，含块缩放开销 |
|---|---:|
| q8_0 | 8.5 |
| q4_0 | 4.5 |
| turbo4_0 | 4.25 |
| turbo3_0 | 3.125 |
| turbo2_0 | 2.125 |

在 K/V 元素数量相同的前提下，`q8_0/q4_0` 比全 q4 的 KV 多占约 44.4%。全 turbo4 比全 q4 仅节省约 5.6% 的 KV 数据；全 turbo3 节省约 30.6%，但模型质量、速度和长上下文可靠性仍需实测。

用户已经报告全 q4 能到约 250K，而此前 ROCmFP4 模型已用 q8/q4 启动到 256K。因此，仅为达到约 256K，没有必要立即承担 TurboQuant 移植和维护成本。此前根据 188K 推算的 288K、391K 等数字只是在固定 KV 预算下的理论比例，不是这些模型的已验证上下文上限。

## 6. 最小运行确认

```bash
curl -s http://127.0.0.1:1234/health
curl -s http://127.0.0.1:1234/props
```

健康检查应返回 `{"status":"ok"}`，随后还应完成一次真实生成请求，确认运行时分配不会 OOM。实际上下文查看 `/props` 的 `default_generation_settings.n_ctx`，不要仅凭启动命令判断自动适配后的有效配置。最大可用长度还需要真实长输入加输出余量的测试。
