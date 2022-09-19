#!/usr/bin/env bash
MODELS="${1:-workspace/mnist/models}"
seq 0 9 | while read -r DIGIT; do
  echo "----"
  stack exec -- mnist recognise --with-batchnorm -M "${MODELS}" "data/mnist/my-digits/mnist-digit-${DIGIT}.png" 2>/dev/null
  echo ""
done
