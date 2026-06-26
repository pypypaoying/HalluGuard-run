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
clone_or_update Nonstationary_Transformers https://github.com/thuml/Nonstationary_Transformers.git c4ec40675d11d50b3d9923657f408d0db6f90f56
clone_or_update TAFAS https://github.com/kimanki/TAFAS.git 139bf980671da4daad728a0fc21d8df508b9203d
clone_or_update Calibration-CDS https://github.com/HALF111/calibration_CDS.git 3c74e5f8b3c921a2e7cda30f36117c295bf8c112
clone_or_update SoP https://github.com/hanyuki23/SoP.git 0bd23d3d0b5497247595b40f86b2d0b095f65eea

echo "Official same-position baseline repos fetched under external/plugin_baselines/."
echo "Pinned source identities are recorded in docs/core_table_manifest.yaml and docs/baseline_repos.tsv."
