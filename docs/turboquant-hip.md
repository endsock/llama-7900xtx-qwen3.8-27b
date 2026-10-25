# Turbo4 KV + DFlash2 / HIP

本分支 `turboquant-hip-dflash2` 保留当前项目的 Qwen3.8、MTP 模型加载和 DFlash2 实现，接入 `/data/project/ai/llama.cpp-turboquant-hip` 中 TurboQuant 的 turbo4 KV 路径。仅移植本次需要的 4-bit PolarQuant、128 元素 WHT、范数修正及原生量化 Flash Attention；没有移植 turbo2、turbo3、InnerQ 校准、权重量化和自适应混合 KV。

## 实现边界

- `-ctk turbo4 -ctv turbo4` 与 `-ctk turbo4_0 -ctv turbo4_0` 等价。
- 每块 128 个值，占 68 字节，含两个 FP16 字段和 64 字节索引，即 4.25 bit/值。`rnorm` 保留为零，与源项目默认的 4-bit PolarQuant 布局一致。
- 量化使用源项目的质心、seed=42 的 WHT 符号和重建范数修正。K/Q 在相同旋转空间做点积，V 的注意力输出进行逆 WHT。
- HIP 原生 Flash Attention 当前支持 K/V 均为 turbo4，head dimension 为 128 或 256；不支持混合 K/V 类型。其他模型和后端未经本机验证。
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
