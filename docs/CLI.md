# CLI Reference

The `netkit` binary is a **desktop development tool** ( **`NETKIT_TARGET=cpu`** only) implemented in C++26 (`src/cli.cpp`, entry via `Cli::Run()` in `src/main.cpp`). It wraps the same inference engine as the library APIs. MCU/MPU builds produce `libnetkit.a` without the CLI — see [BUILD_TARGETS.md](BUILD_TARGETS.md).

```bash
make              # NETKIT_TARGET=cpu (default) — builds ./netkit
./netkit <command> [arguments]
./netkit help     # print usage (also: -h, --help)
```

A C API equivalent exists as `nk_cli_run(argc, argv)` for embedding the same command dispatch.

Convert ONNX to `.nk` with the Python packager — see [python/README.md](../python/README.md) and [NK_FORMAT.md](NK_FORMAT.md). Embed a `.nk` in firmware with `python -m netkit aot` — see [GETTING_STARTED.md](GETTING_STARTED.md#5-aot-compile-embed-nk-in-firmware).

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
| `run` | `--input <values>` | `<model.nk>` |
| `inspect` | `--full` | `<model.nk>` |

## Commands

### `test`

Run the full C++ API regression suite (same cases as `make test-cpp`).

```bash
./netkit test
```

Exit code `0` if all cases pass, `1` if any fail.

Prints per-case PASS/FAIL lines and a summary:

```
Passed: 73
Failed: 0
```

### `run`

Load a model and run one forward pass.

```bash
./netkit run <model.nk> --input <values>
```

**Options:**

| Option | Form | Description |
|--------|------|-------------|
| `--input` | `--input 1,2,3` or `--input=1,2,3` | Comma- or space-separated float32 values |

**Examples:**

```bash
# MLP: input shape [1, 2]
./netkit run models/test_mlp.nk --input 1,2

# CNN: input shape [4, 4, 1] — sixteen values
./netkit run models/cnn_4x4_single.nk --input=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

**Behavior:**

1. Parses the `.nk` header and prints a boxed network summary
2. Loads weights from the same file
3. Validates input element count against the model input shape
4. Runs forward pass using a **model-sized heap arena** (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN)
5. Prints labeled input and output tensors

**Input count:**

| Network | Required values |
|---------|----------------|
| MLP | `batch × features` (product of input shape) |
| CNN | `H × W × C` (NHWC flatten order) |

Maximum 4096 input floats per invocation.

**Errors:** Missing `--input`, wrong value count, parse/load failures, or arena overflow print to stderr and return exit code `1`.

### `inspect`

Pretty-print the model architecture as a boxed network summary.

```bash
./netkit inspect <model.nk>
./netkit inspect <model.nk> --full
```

**Default output:** Network Summary block with name, type, version, input shape, and a numbered layer list.

Example (`./netkit inspect models/mnist_cnn.nk`):

```
=====================================================
Network Summary
=====================================================

Name        : mnist_cnn
Type        : CNN
Version     : 1

Input Shape : [28, 28, 1]

Layers (7)
-----------------------------------------------------
  [0] Conv2D kernel=3 stride=1 filters=32 activation=relu
  ...
```

The C API equivalent is `nk_arch_print()`.

**`--full`:** Load weights, run a zero-input forward pass, and report arena memory usage after load and forward. Use this to size embedded arena buffers before deployment. The C API equivalent is `nk_inspect_model()`.

On the default **CPU (heap arena)** build, the CLI allocates a model-appropriate arena automatically (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN). Examples use **`NK_ARENA_DEFAULT_CAPACITY` (4 MiB)**. Build with `NETKIT_GLOBAL_ARENA=1` for static backing — see [BUILD_TARGETS.md](BUILD_TARGETS.md).

## Path resolution

If `<model.nk>` is not found in the current directory, the CLI tries `../<model.nk>`. Run from the repo root or ensure model paths are reachable.

## Relationship to examples

| Tool | Interface | Typical use |
|------|-----------|-------------|
| `./netkit run ...` | CLI | Quick one-off inference, scripting |
| `./examples/infer_cpp` | C++26 API | Reference for native integration |
| `./examples/infer_c` | C23 API | Reference for embedded C integration |

The examples take input as separate argv floats instead of `--input`:

```bash
./examples/infer_cpp models/test_mlp.nk 1 2
./examples/infer_c models/test_mlp.nk 1 2
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for build and link instructions.

## Prerequisites

The CLI exists only in **CPU** builds:

```bash
make              # default: NETKIT_TARGET=cpu
make cpu          # same
```

MCU/MPU builds (`make NETKIT_TARGET=mcu lib`) do **not** produce `./netkit`. Use the library API on device; use the desktop CLI to develop and size arenas.

## Build and memory (CLI)

| Setting | Default | Override |
|---------|---------|----------|
| Target | `NETKIT_TARGET=cpu` | — |
| Arena backing | **Heap** (`malloc`) | `NETKIT_GLOBAL_ARENA=1` → static 4 MiB buffer in CLI |
| Arena size (heap) | **Model-based** — see below | Not user-configurable at CLI argv level |

CLI arena sizes (via `ArenaUtil::CapacityForInputElements`):

| Model class | Heap size |
|-------------|-----------|
| Hand MLP/CNN | 64 KiB |
| MNIST MLP (784 inputs) | 2 MiB |
| MNIST CNN (784 inputs) | 4 MiB |

C API embed: `nk_cli_run(argc, argv)` — requires `NETKIT_DESKTOP` (CPU build).

See [BUILD_TARGETS.md](BUILD_TARGETS.md) and [ARENA.md](ARENA.md).
