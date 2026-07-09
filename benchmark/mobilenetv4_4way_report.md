# MobileNetV4-small 4-way benchmark (host/CPU)

Methodology matches `benchmark/netkit/src/mobilenetv4_main.cc` and `benchmark/tflm/src/mobilenetv4_*_main.cc`: 10 MNIST-derived inputs (28×28×1 upsampled to 56×56×3), 30 loops, 300 timed invokes. Primary metric: **warm median** invoke latency (µs).

| Runtime | dtype   | warm median (ms) | warm min (ms) | vs netkit f32 |
|---------|---------|------------------|---------------|---------------|
| netkit  | float32 | 2.45             | 2.39          | 1.00×         |
| netkit  | int8    | 2.31             | 2.11          | 0.94× (faster)|
| TFLM    | float32 | 14.94            | 13.62         | 6.09× slower  |
| TFLM    | int8    | 19.20            | 17.70         | 7.83× slower  |

Netkit int8 is ~6% faster than netkit float32 on this host run (UIB composite quant path + CMSIS-NN kernels).

## Run

```bash
./tools/run_mobilenetv4_4way_benchmark.sh
# writes benchmark/mobilenetv4_4way_results.txt
```

Prerequisites:

- `models/mobilenetv4_small_int8.nk` (`python3 tools/write_mobilenetv4_small_int8.py`)
- TFLM int8 export (`make -C benchmark/tflm export-mobilenetv4-int8`) — uses `benchmark/tflm/tools/run_export_mobilenetv4_int8.sh` (TF 2.16 venv) when system TensorFlow is broken

Individual targets:

```bash
make -C benchmark/netkit run-mobilenetv4
make -C benchmark/netkit run-mobilenetv4-int8
make -C benchmark/tflm run-mobilenetv4
make -C benchmark/tflm run-mobilenetv4-int8
```
