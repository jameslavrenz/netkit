# CLI Reference

The `netkit` binary is a **desktop development tool** implemented in C++26 (`src/cli.cpp`, entry via `Cli::Run()` in `src/main.cpp`). It wraps the same engine as the library APIs.

```bash
make              # builds ./netkit
./netkit <command> [arguments]
```

A C API equivalent exists as `nk_cli_run(argc, argv)` for embedding the same command dispatch.

## Commands

### `test`

Run the full C++ API vectors regression suite (same cases as `make test-cpp`).

```bash
./netkit test
```

Exit code `0` if all cases pass, `1` if any fail.

Prints per-case PASS/FAIL lines and a summary:

```
Passed: 8
Failed: 0
```

### `run`

Load a model and run one forward pass.

```bash
./netkit run <model.json> --input <values>
```

**Options:**

| Option | Form | Description |
|--------|------|-------------|
| `--input` | `--input 1,2,3` or `--input=1,2,3` | Comma- or space-separated float32 values |

**Examples:**

```bash
# MLP: input shape [1, 2]
./netkit run models/test_mlp.json --input 1,2

# CNN: input shape [4, 4, 1] — sixteen values
./netkit run models/cnn_4x4_single.json --input=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

**Behavior:**

1. Parses architecture from `<model.json>` and loads companion `.bin` weights
2. Validates input element count against the model's `input` shape
3. Runs forward pass using the internal 64 KiB arena
4. Prints labeled input and output tensors

**Input count:**

| Network | Required values |
|---------|----------------|
| MLP | `batch × features` (product of `input` array in JSON) |
| CNN | `H × W × C` (NHWC flatten order) |

Maximum 4096 input floats per invocation.

**Errors:** Missing `--input`, wrong value count, parse/load failures, or arena overflow print to stderr and return exit code `1`.

### `inspect`

Print architecture metadata, weight file summary, and arena memory usage.

```bash
./netkit inspect <model.json>
```

**Output includes:**

- JSON fields: version, network kind, input shape, layer list, expected weight float count
- Weight file path, loaded float count, match/mismatch vs architecture
- Arena bytes used after load and after a zero-input forward pass, plus remaining capacity

Use this to size embedded arena buffers before deployment. The C API equivalent is `nk_inspect_model()`.

## Path resolution

If `<model.json>` is not found in the current directory, the CLI tries `../<model.json>`. Run from the repo root or ensure model paths are reachable.

## Relationship to examples

| Tool | Interface | Typical use |
|------|-----------|-------------|
| `./netkit run ...` | CLI | Quick one-off inference, scripting |
| `./examples/infer_cpp` | C++26 API | Reference for native integration |
| `./examples/infer_c` | C23 API | Reference for embedded C integration |

The examples take input as separate argv floats instead of `--input`:

```bash
./examples/infer_cpp models/test_mlp.json 1 2
./examples/infer_c models/test_mlp.json 1 2
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for build and link instructions.
