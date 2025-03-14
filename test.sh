#!/usr/bin/env bash
set -euo pipefail

nim c -r nim_micrograd/tests/test_nn.nim
nim c -r nim_micrograd/tests/test_engine.nim

nim c -r makemore-exercise/notebook.nim