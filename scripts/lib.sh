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
  local install_url="https://astral.sh/uv/install.sh"
  local min_version="0.9.18"

  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "$install_url" | sh; then
      error "Failed to install uv"
      return 1
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi

  local uv_version
  uv_version="$(uv --version 2>/dev/null | awk '{print $2}')"
  if [ -z "$uv_version" ]; then
    error "Unable to determine uv version"
    return 1
  fi
  if ! version_ge "$uv_version" "$min_version"; then
    error "uv $uv_version is older than required $min_version"
    error "Please upgrade uv (e.g., 'brew upgrade uv' or"
    error "'curl -LsSf https://astral.sh/uv/install.sh | sh' and ensure ~/.local/bin is in PATH)."
    return 1
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
