# Vendored license texts

Copies of third-party license files for components used by netkit (runtime
backends, Python tooling, and peer-benchmark stacks). See
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for roles, copyright
holders, and redistribution notes.

These files are mirrored so attribution remains available when large fetch trees
(`third_party/XNNPACK/`, `third_party/NMSIS/`, `third_party/ESP-NN/`, TFLM
downloads, ORT / TVM checkouts) are not present.

**Refresh runtime texts** after bumping pins:

```bash
./tools/sync_third_party_licenses.sh
# also run automatically at the end of tools/fetch_*.sh
```

| Prefix / path | Source |
|---------------|--------|
| `CMSIS-*` | Runtime — Arm CMSIS-Core / CMSIS-NN (git submodules) |
| `ESP-NN*` | Runtime — Espressif ESP-NN (`make esp-nn-init`) |
| `NMSIS*` | Runtime — Nuclei NMSIS / NMSIS-NN (`make nmsis-init`) |
| `XNNPACK*`, `pthreadpool*`, `cpuinfo*`, `clog*`, `FXdiv*`, `KleidiAI*` | Runtime — XNNPACK + CMake deps (`make xnnpack-init`) |
| `numpy*`, `onnx*`, `onnxruntime*`, `torch*`, `timm*`, `onnxscript*` | Python tooling (`python/pyproject.toml`) |
| `LiteRT*`, `tflite-micro*`, `flatbuffers*`, `gemmlowp*`, `ruy*`, `tflm-bundled-*` | Peer benches (LiteRT wheel / TFLM) |
| `apache-tvm.Apache-2.0.txt`, `apache-tvm.NOTICE.txt` | Apache TVM / microTVM peer |
| `apache-tvm/` | Upstream TVM `licenses/` bundle |
| `apache-tvm-3rdparty-dmlc-core*` | TVM `3rdparty/dmlc-core` |
