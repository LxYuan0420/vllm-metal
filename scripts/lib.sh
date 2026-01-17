#!/bin/bash
# Common library functions for vllm-metal scripts

# Print an error message
error() {
  echo -e "Error: $*" >&2
}

# Print a success message
success() {
  echo -e "✓ $*"
}

# Print a section header
section() {
  echo "=== $* ==="
}

# Check if running on Apple Silicon
is_apple_silicon() {
  [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]
}

# Exit with a clear error message when running on unsupported platforms
require_apple_silicon() {
  if ! is_apple_silicon; then
    error "This script requires Apple Silicon macOS (Darwin arm64)."
    return 1
  fi
}

# Compare dotted versions (return 0 if $1 >= $2)
version_ge() {
  local IFS=.
  local -a v1=($1) v2=($2)
  local i

  for ((i=${#v1[@]}; i<${#v2[@]}; i++)); do
    v1[i]=0
  done
  for ((i=${#v2[@]}; i<${#v1[@]}; i++)); do
    v2[i]=0
  done

  for ((i=0; i<${#v1[@]}; i++)); do
    if ((10#${v1[i]} > 10#${v2[i]})); then
      return 0
    fi
    if ((10#${v1[i]} < 10#${v2[i]})); then
      return 1
    fi
  done
  return 0
}

# Ensure uv is installed
ensure_uv() {
  local install_url="${VLLM_METAL_UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
  local install_version="${VLLM_METAL_UV_VERSION:-}"
  local min_version="${VLLM_METAL_UV_MIN_VERSION-0.9.0}"

  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if [ -n "$install_version" ]; then
      if ! UV_VERSION="$install_version" curl -LsSf "$install_url" | sh; then
        error "Failed to install uv"
        return 1
      fi
    else
      if ! curl -LsSf "$install_url" | sh; then
        error "Failed to install uv"
        return 1
      fi
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [ -n "$min_version" ]; then
    local uv_version
    uv_version="$(uv --version 2>/dev/null | awk '{print $2}')"
    if [ -z "$uv_version" ]; then
      error "Unable to determine uv version"
      return 1
    fi
    if ! version_ge "$uv_version" "$min_version"; then
      error "uv $uv_version is older than required $min_version"
      error "Set VLLM_METAL_UV_INSTALL_URL or VLLM_METAL_UV_VERSION to upgrade"
      return 1
    fi
  fi
}

# Ensure virtual environment exists and is activated
ensure_venv() {
  if [ ! -d "$1" ]; then
    section "Creating virtual environment"
    uv venv "$1" --clear --python 3.12
  fi

  # shellcheck source=/dev/null
  source "$1/bin/activate"
}

# Install dev dependencies
install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]"
}

# Full development environment setup
setup_dev_env() {
  ensure_uv
  ensure_venv ".venv-vllm-metal"
  install_dev_deps
}

# Get version from pyproject.toml
get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}
