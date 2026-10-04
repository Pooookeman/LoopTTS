#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/third_party/EmoVoice"
patch_file="$repo_root/inference/patches/emovoice-refiner-inference.patch"
upstream_commit="5285cb891611cf1ee2d9bd07b931cd3cf967cd64"

if [[ ! -d "$upstream_dir/.git" ]]; then
  mkdir -p "$repo_root/third_party"
  git clone https://github.com/yanghaha0908/EmoVoice.git "$upstream_dir"
  git -C "$upstream_dir" checkout --detach "$upstream_commit"
fi

if [[ "$(git -C "$upstream_dir" rev-parse HEAD)" != "$upstream_commit" ]]; then
  echo "EmoVoice checkout differs from the pinned version; use a clean checkout at $upstream_commit." >&2
  exit 1
fi

if git -C "$upstream_dir" apply --reverse --check "$patch_file"; then
  echo "Refiner patch is already applied."
elif git -C "$upstream_dir" apply --check "$patch_file"; then
  git -C "$upstream_dir" apply "$patch_file"
  echo "Refiner patch applied."
else
  echo "Could not apply Refiner patch to this EmoVoice checkout." >&2
  exit 1
fi
