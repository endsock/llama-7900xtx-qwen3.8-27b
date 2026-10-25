# turbo4_0 vs q8_0 KV 对照结果

| ID | 类型 | q8_0 | turbo4_0 | q8 秒 | turbo4 秒 | q8 prompt tok | turbo4 prompt tok |
|---|---|---:|---:|---:|---:|---:|---:|
| t0-1-mul | t0 | PASS | PASS | 1.8 | 1.6 | 67 | 67 |
| t0-2-perim | t0 | PASS | PASS | 1.6 | 1.2 | 72 | 72 |
| t0-3-moon | t0 | PASS | PASS | 1.3 | 1.3 | 62 | 62 |
| t0-4-dup | t0 | PASS | PASS | 1.7 | 1.9 | 77 | 77 |
| t0-5-age | t0 | PASS | PASS | 1.6 | 1.9 | 75 | 75 |
| t1-l0/d0 | t1 | PASS | PASS | 7.3 | 8.4 | 1983 | 1983 |
| t1-l0/d1 | t1 | PASS | PASS | 6.9 | 6.8 | 1983 | 1983 |
| t1-l0/d2 | t1 | PASS | PASS | 7.2 | 7.1 | 1981 | 1981 |
| t1-l1/d0 | t1 | PASS | PASS | 13.9 | 17.0 | 7653 | 7653 |
| t1-l1/d1 | t1 | PASS | PASS | 14.2 | 17.8 | 7653 | 7653 |
| t1-l1/d2 | t1 | PASS | PASS | 23.4 | 16.5 | 7652 | 7652 |
| t1-l2/d0 | t1 | PASS | PASS | 23.8 | 32.4 | 15212 | 15212 |
| t1-l2/d1 | t1 | PASS | PASS | 23.2 | 34.9 | 15211 | 15211 |
| t1-l2/d2 | t1 | PASS | PASS | 23.4 | 32.7 | 15211 | 15211 |
| t1-l3/d0 | t1 | PASS | PASS | 45.5 | 82.3 | 30308 | 30308 |
| t1-l3/d1 | t1 | PASS | PASS | 47.1 | 82.9 | 30311 | 30311 |
| t1-l3/d2 | t1 | PASS | PASS | 45.7 | 86.2 | 30309 | 30309 |
| t1-l4/d0 | t1 | PASS | PASS | 93.8 | 210.0 | 54883 | 54883 |
| t1-l4/d1 | t1 | PASS | PASS | 95.4 | 212.2 | 54886 | 54886 |
| t1-l4/d2 | t1 | PASS | PASS | 94.4 | 211.7 | 54886 | 54886 |
| t2-1-max | t2 | FAIL | FAIL | 111.7 | 219.8 | 52451 | 52451 |
| t2-2-count | t2 | FAIL | FAIL | 19.9 | 24.7 | 52446 | 52446 |
| t2-3-earliest | t2 | FAIL | FAIL | 21.7 | 26.9 | 52450 | 52450 |
| t2-4-lookup | t2 | PASS | PASS | 4.1 | 6.4 | 52445 | 52445 |

## 汇总

- t0: q8_0 5/5，turbo4_0 5/5
- t1: q8_0 15/15，turbo4_0 15/15
- t2: q8_0 1/4，turbo4_0 1/4

## 失败项目答案

### t2-1-max

- 期望：`EMP-0311`; `39852`
- q8_0：``
- turbo4_0：``

### t2-2-count

- 期望：`88`
- q8_0：``
- turbo4_0：``

### t2-3-earliest

- 期望：`EMP-0341`
- q8_0：``
- turbo4_0：``

