#!/bin/bash
# 在当前设备上用 trtexec 将 ONNX 模型转换为 TensorRT engine
# TensorRT engine 不能跨设备/跨版本使用，必须在目标设备(Jetson/Orin)上重新生成
#
# 用法:
#   ./onnx2engine.sh                  # 转换 assets/ 下所有 .onnx (FP16)
#   ./onnx2engine.sh assets/0526.onnx # 只转换指定模型 (FP16)
#   ./onnx2engine.sh --fp32 [xxx.onnx] # 使用 FP32 精度转换

set -e

PRECISION="--fp16"
if [ "$1" == "--fp32" ]; then
  PRECISION=""
  shift
fi

# 定位 trtexec
TRTEXEC=$(which trtexec 2>/dev/null || true)
if [ -z "$TRTEXEC" ]; then
  for p in /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec; do
    if [ -x "$p" ]; then
      TRTEXEC="$p"
      break
    fi
  done
fi
if [ -z "$TRTEXEC" ]; then
  echo "[ERROR] 找不到 trtexec，请确认已安装 TensorRT"
  exit 1
fi
echo "[INFO] 使用 trtexec: $TRTEXEC"

convert() {
  local onnx="$1"
  local engine="${onnx%.onnx}.engine"
  echo "[INFO] 转换: $onnx -> $engine"
  "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" $PRECISION
  echo "[INFO] 完成: $engine"
}

if [ $# -gt 0 ]; then
  # 转换指定的 onnx 文件
  for onnx in "$@"; do
    if [ ! -f "$onnx" ]; then
      echo "[ERROR] 文件不存在: $onnx"
      exit 1
    fi
    convert "$onnx"
  done
else
  # 默认转换 assets/ 下所有 onnx
  shopt -s nullglob
  onnx_files=(assets/*.onnx)
  if [ ${#onnx_files[@]} -eq 0 ]; then
    echo "[ERROR] assets/ 下没有找到 .onnx 文件"
    exit 1
  fi
  for onnx in "${onnx_files[@]}"; do
    convert "$onnx"
  done
fi

echo "[INFO] 全部转换完成"
