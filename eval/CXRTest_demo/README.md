# CXRTest_demo

- `jsons/*_for_infer_normalize_demo.json` — 各测试集前 10 条样本；`images` 为相对于 `jsons/` 的路径（如 `../CXRTest/...`）。
- `CXRTest/` — 上述 JSON 引用的影像目录（与原始 `CXRTest/...` 布局一致）。

加载图片时请从 `jsons/` 解析相对路径，或将工作目录设为 `jsons/` 再拼接路径。
