# Vectors Regression Tests

netkit validates inference with **declarative test files**: `*.vectors.json` paired with model bundles. Both test suites (`make test-cpp` and `make test-c`) exercise the same vector models.

## File format

```json
{
  "model": "test_mlp.json",
  "cases": [
    {
      "name": "2-layer forward",
      "input": [1, 2],
      "expected": [3, 3]
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `model` | Path to the architecture JSON, relative to the vectors file's directory |
| `cases` | Array of test cases |
| `cases[].name` | Human-readable label (optional; defaults to `"case"`) |
| `cases[].input` | Flat float32 input array; length must match model input element count |
| `cases[].expected` | Flat float32 expected output; length must match model output element count |

All values are parsed as **float32** (`strtof` internally).

## Comparison tolerance

Outputs are compared element-wise with absolute tolerance **1e-5**:

```
|actual[i] - expected[i]| <= 1e-5
```

On mismatch, the runner prints expected vs actual and the failing index.

## Running tests

```bash
make test         # C++ API tests, then C API tests
make test-cpp     # ./netkit test  (C++26)
make test-c       # ./tests/test_c_api  (C23)
```

The C++ suite is implemented in `src/test.cpp` via `VectorsLoader::RunVectorsFile`. The C suite calls `nk_run_all_tests()` which delegates to the same C++ runner.

## Registered vector files

These eight files are wired in `src/test.cpp`:

| Vectors file | Model | Network |
|--------------|-------|---------|
| `models/test_mlp.vectors.json` | `test_mlp.json` | MLP |
| `models/mlp_hand.vectors.json` | `mlp_hand.json` | MLP |
| `models/test_cnn.vectors.json` | `test_cnn.json` | CNN |
| `models/cnn_4x4_single.vectors.json` | `cnn_4x4_single.json` | CNN |
| `models/cnn_hand.vectors.json` | `cnn_hand.json` | CNN |

(`mlp_hand` and `cnn_hand` each contain multiple cases in one file.)

## Adding a new regression case

1. Create or update the model bundle (`model.json` + `model.bin`). See [MODEL_FORMAT.md](MODEL_FORMAT.md).
2. Add a `model.vectors.json` with at least one case.
3. Register the vectors file in `src/test.cpp` inside `run_all_tests()`:

```cpp
merge(VectorsLoader::RunVectorsFile("models/my_model.vectors.json"));
```

4. Run `make test` — both C++ and C suites pick up the change automatically (C suite calls the same `run_all_tests` via the C API bridge).

## Input layout reminders

| Network | Flat `input` order |
|---------|-------------------|
| MLP | Row-major `[batch, features]` |
| CNN | NHWC `[height, width, channels]` |

For a 2×2×2 CNN input, flatten as `[h0w0c0, h0w0c1, h0w1c0, h0w1c1, h1w0c0, ...]`.

## C API entry points

```c
nk_test_summary_t nk_run_vectors_file(const char* vectors_path);
nk_test_summary_t nk_run_all_tests(void);
```

Returns `{ passed, failed }` counts. Used by `tests/test_c_api.c` for the vectors portion of the C regression suite.
