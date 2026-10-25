# Turbo4 KV + DFlash2 / HIP

本分支 `turboquant-hip-dflash2` 保留当前项目的 Qwen3.8、MTP 模型加载和 DFlash2 实现，接入 `/data/project/ai/llama.cpp-turboquant-hip` 中 TurboQuant 的 turbo4 KV 路径。仅移植本次需要的 4-bit PolarQuant、128 元素 WHT、范数修正及原生量化 Flash Attention；InnerQ 使用本分支的离线两阶段校准，没有移植 turbo2、turbo3、权重量化和自适应混合 KV。

## 实现边界

- `-ctk turbo4 -ctv turbo4` 与 `-ctk turbo4_0 -ctv turbo4_0` 等价。
- 每块 128 个值，占 68 字节，含两个 FP16 字段和 64 字节索引，即 4.25 bit/值。`rnorm` 保留为零，与源项目默认的 4-bit PolarQuant 布局一致。
- 量化使用源项目的质心、seed=42 的 WHT 符号和重建范数修正。K/Q 在相同旋转空间做点积，V 的注意力输出进行逆 WHT。
- HIP 原生 Flash Attention 当前支持双 turbo4，以及 `q8_0 K + turbo4_0 V` 混合组合，head dimension 为 128 或 256。InnerQ 配置目前仅支持双 turbo4。其他模型和后端未经本机验证。
- 类型 ID 使用本项目空闲的 `102`，没有复用源项目与本项目 Q2_0 冲突的 `42`。本次仅用于运行时 KV，不用于生成 TurboQuant 权重 GGUF。
- 保留 `LLAMA_ATTN_ROT_DISABLE=1`、单槽位、非统一 KV、物理批大小 256 和文档中的 DFlash2/ngram 参数。

主要源码：

| 路径 | 内容 |
|---|---|
| `ggml/src/ggml-turbo4.{c,h}` | CPU 参考量化、反量化、旋转常量和 WHT |
| `ggml/src/ggml-cuda/turbo4.{cu,cuh}` | HIP/CUDA WHT 和 KV 写入 |
| `ggml/src/ggml-cuda/fattn-{common,vec}.cuh` | 原生 turbo4 KQ 点积和 V 解码 |
| `src/llama-graph.cpp` | Q 旋转和注意力输出逆旋转 |
| `tests/test-turbo4.cpp` | CPU/HIP 量化与注意力数值测试 |

## InnerQ 离线校准

InnerQ 先统计每层 K 在 128 元素组内各通道的 RMS，计算 `s = clamp((mean_rms / rms)^strength, 0.5, 2)`，保存 `scale_inv = 1 / s`。固定配置下，K/V 写入前乘 s，Q 正向 WHT 前乘 scale_inv，V 输出逆 WHT 后乘 scale_inv。默认 strength 为 0.5，可用 `TURBO_INNERQ_STRENGTH` 设置为 (0, 1] 内的值。

校准过程不施加缩放，也不在请求中途启用缩放。校准服务正常退出时保存每层配置，正式服务在创建 KV 缓存时加载配置，整个生命周期保持不变。DFlash2 共享 KV 的层共用对应的 scale，独立层分别统计。旧的 `TURBO_INNERQ=N` 在线切换入口已移除，设置该变量会报错。

```bash
mkdir -p results/innerq-profile
TURBO_INNERQ_CALIBRATE="$PWD/results/innerq-profile" CTX=65536 bash scripts/serve-turbo4-dflash2.sh
# Send representative requests, then terminate the server normally with SIGTERM.
TURBO_INNERQ_PROFILE="$PWD/results/innerq-profile" bash scripts/serve-turbo4-dflash2.sh
```

两个变量互斥，不设置时保持原 turbo4 行为。目前配置只支持双 turbo4 KV；128/256 维 head 使用相同的 128 元素分组。配置按模型标识、层号和 SWA 范围区分，文件缺失或 scale 非法时报错，不回退为未校准配置。统计数量是 K 量化块数，不是输入 token 数。

清空 KV 数据后重新安装同一份固定 scale。保存 KV 状态时同时保存 scale，恢复时拒绝不匹配的配置。不要将不同模型权重或不同校准配置下的缓存混用。开启 InnerQ 不降低 KV 的 4.25 bit/值，也不保证质量或吞吐提升，仍需代表性任务验证。

`scripts/kv-compare/run_innerq_snake.py` 在 1234 端口空闲时完成校准、正常退出、加载配置和 65536 上下文贪吃蛇测速，成功后保留该 InnerQ 服务。结果写入 `results/snake-turbo4-innerq.json`，运行日志在 `/tmp/innerq-calibration.log` 和 `/tmp/innerq-snake.log`。

## 编译

```bash
cmake -S . -B build-turbo4 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_HIP_NO_VMM=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-10.1 \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-turbo4 --target llama-server test-turbo4 -j 8

./build-turbo4/bin/test-turbo4
./build-turbo4/bin/test-turbo4 --gpu
```

本机 CPU/HIP 的 128/256 head dimension、I32/I64 逆序行索引和零向量均通过测试。HIP 注意力结果与 CPU 参考计算的最大绝对误差约 `3.95e-4`；WHT 正逆变换最大绝对误差约 `2.39e-7`。

## 启动

```bash
bash scripts/serve-turbo4-dflash2.sh
```

脚本完整保留 `docs/llm_start.md` 中的 DFlash2/ngram 参数，将 KV 改为 `turbo4_0/turbo4_0`、上下文改为 `262144`，服务地址为 `http://0.0.0.0:1234`。

在当前服务器逻辑下，槽位上下文不能超过模型训练长度 `262144`。实测传入 `-c 327680` 会分配更多显存，但 `tools/server/server-context.cpp` 会将槽位截回 `262144`，`/props` 同样报告 `262144`，因此它不算有效的 320K 上下文。此配置普通请求后显存达到 24543 MiB，不能作为日常配置。

`-c 262144` 已通过以下测试：

| 请求 | 结果 |
|---|---|
| 5008 token 输入 | 正常生成，DFlash 草稿接受 6/17 token |
| 42108 token 输入 | 预填充约 132.52 秒，正常生成，DFlash 草稿接受 5/14 token，无 OOM |
| OpenAI 聊天接口：17 + 25 | 正确回答 `42`，DFlash 草稿接受 28/41 token |

加载显存 22975 MiB，42108 token 请求后 23513 MiB，总显存 24560 MiB。最终健康检查返回 `{"status":"ok"}`。测试中的输出较短且输入为合成记录，不将这些吞吐和接受率作为固定性能保证。

本记录不是满 256K 输入压力测试，也不保证所有长文本上的量化质量。当前留下的服务 PID 为 `13960`；PID 仅用于本次会话，重启后会变化。

编译和测试日志：`/tmp/dflash-turbo4-build.log`、`/tmp/dflash-turbo4-tests.log`。运行日志：`/tmp/llama-turbo4-262144.log`。临时目录中的日志重启后可能丢失。
