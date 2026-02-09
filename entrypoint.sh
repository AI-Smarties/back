#!/bin/sh

set -eu

fastapi run src/main.py --host 0.0.0.0
