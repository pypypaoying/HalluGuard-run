#!/usr/bin/env bash
set -euo pipefail

mkdir -p external/plugin_baselines

clone_or_update() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local dst="external/plugin_baselines/${name}"
  if [[ ! -d "${dst}/.git" ]]; then
    git clone "${url}" "${dst}"
  fi
  git -C "${dst}" fetch --all --tags
  git -C "${dst}" checkout "${commit}"
}

clone_or_update RevIN https://github.com/ts-kim/RevIN.git fee40bc6c87cb536d048bcf1c14c4ed644b875e1
clone_or_update Dish-TS https://github.com/weifantt/Dish-TS.git e674d3b94b832491f63a533d60e40a75031d2c75
clone_or_update SAN https://github.com/icantnamemyself/SAN.git 7e1ca66251a91a89290846b310145c5f5db3ffc3

echo "Verified plug-in repos fetched under external/plugin_baselines/."
echo "See docs/baseline_repos.tsv for unverified sources that still need confirmation."
