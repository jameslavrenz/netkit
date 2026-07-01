# CLI Reference

The `netkit` binary is a **desktop development tool** implemented in C++26 (`src/cli.cpp`, entry via `Cli::Run()` in `src/main.cpp`). It wraps the same engine as the library APIs.

```bash
make              # builds ./netkit
./netkit <command> [arguments]
./netkit help     # print usage (also: -h, --help)
```

A C API equivalent exists as `nk_cli_run(argc, argv)` for embedding the same command dispatch.

## Global options

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Print command usage and exit (exit code `0`) |
| `help` | Same as `-h` / `--help` when used as the command or flag |

Examples:

```bash
./netkit --help
./netkit help
./netkit run --help          # global help (any position after argv[0])
```

Running `./netkit` with no arguments prints the same help text and exits with code `1`.

## Command options summary

| Command | Options | Required arguments |
|---------|---------|-------------------|
| `test` | — | — |
| `run` | `--input <values>` | `<model.json>` |
| `inspect` | `--full` | `<model.json>` |

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

1. Parses architecture from `<model.json>` and prints a boxed network summary
2. Loads companion `.bin` weights
3. Validates input element count against the model's `input` shape
4. Runs forward pass using the internal 64 KiB arena
5. Prints labeled input and output tensors

**Input count:**

| Network | Required values |
|---------|----------------|
| MLP | `batch × features` (product of `input` array in JSON) |
| CNN | `H × W × C` (NHWC flatten order) |

Maximum 4096 input floats per invocation.

**Errors:** Missing `--input`, wrong value count, parse/load failures, or arena overflow print to stderr and return exit code `1`.

### `inspect`

Pretty-print the model architecture as a boxed network summary.

```bash
./netkit inspect <model.json>
./netkit inspect <model.json> --full
```

**Default output:** Network Summary block with name, type, version, input shape, and a numbered layer list.

Example (`./netkit inspect models/mnist_cnn.json`):

```
=====================================================
Network Summary
=====================================================

Name        : mnist_cnn
Type        : CNN
Version     : 1

Input Shape : 28 x 28 x 1

Layers (7)
-----------------------------------------------------
 0  Conv2D      32 filters   3x3  stride=1  ReLU
 1  MaxPool2D   2x2           stride=2
 2  Conv2D      64 filters   3x3  stride=1  ReLU
 3  MaxPool2D   2x2           stride=2
 4  Flatten
 5  Dense       128 units  ReLU
 6  Dense       10 units  Softmax

=====================================================
```

The C API equivalent is `nk_arch_print()`.

**`--full`:** Legacy diagnostic mode — compact architecture dump, weight file summary, and arena memory usage after load and a zero-input forward pass. Use this to size embedded arena buffers before deployment. The C API equivalent for arena sizing is `nk_inspect_model()`.

The CLI uses a **64 KiB** internal arena (`Arena::kDefaultCapacity`). That is enough for hand test models. **MNIST models** need multi-MiB buffers — `inspect --full` on `mnist_mlp.json` / `mnist_cnn.json` may fail with arena overflow on the CLI even though `make test` passes (tests use 2 MiB / 4 MiB). Size your own firmware buffer from `nk_inspect_model()` with a large enough arena, or see [ARENA.md](ARENA.md).

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
