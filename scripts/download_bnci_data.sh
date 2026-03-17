#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output_dir>" >&2
  exit 1
fi

out_dir="$1"
mkdir -p "$out_dir"

base_url="https://bnci-horizon-2020.eu/database/data-sets/001-2014"

for i in {1..9}; do
  for suffix in E T; do
    file="A0${i}${suffix}.mat"
    url="${base_url}/${file}"
    echo "Downloading ${url}"
    curl -fL -o "${out_dir}/${file}" "${url}"
  done
done

echo "Done. Files saved to ${out_dir}"
