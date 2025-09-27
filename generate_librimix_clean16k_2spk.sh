#!/usr/bin/env bash

set -euo pipefail

outdir=${1:? "Give output dir, e.g.  /data/librimix16k"}
librispeech_root="$outdir/LibriSpeech"
repo_root="$(dirname "$0")"
mkdir -p "$outdir" "$librispeech_root"

# --- download three clean splits ------------------------------------------------
#download () { url=$1; tgt=$2; [ -f "$tgt" ] || wget -q -O "$tgt" "$url"; \
#              [ -d "${tgt%.tar.gz}" ] || tar -xf "$tgt" -C "$librispeech_root"; }

download () {
  url=$1
  tgt=$2
  [ -e "$tgt" ] || { echo " › downloading $url"; wget -q -O "$tgt" "$url"; }
  [ -d "${tgt%.tar.gz}" ] || { echo " › extracting $(basename "$tgt")"; tar -xf "$tgt" -C "$librispeech_root"; }
}

download https://www.openslr.org/resources/12/dev-clean.tar.gz        "$librispeech_root/dev-clean.tar.gz"
download https://www.openslr.org/resources/12/test-clean.tar.gz       "$librispeech_root/test-clean.tar.gz"
download https://www.openslr.org/resources/12/train-clean-100.tar.gz  "$librispeech_root/train-clean-100.tar.gz"

# --- regenerate from scratch (remove old partial dir first) ---------------------
rm -rf "$outdir/Libri2Mix"

# --- move the extracted data to the correct file for the other script to work ----------
mv dataset/LibriSpeech/LibriSpeech/* dataset/LibriSpeech/


python3 "$repo_root/scripts/create_librimix_from_metadata_new.py" \
       --librispeech_dir "$librispeech_root" \
       --metadata_dir "$repo_root/metadata/Libri2Mix" \
       --librimix_outdir "$outdir" \
       --n_src 2 \
       --freqs 16k \
       --modes min \
       --wham_dir "" \
       --types mix_clean
        
#python "scripts/create_librimix_from_metadata_new.py" \
#       --librispeech_dir "dataset/LibriSpeech" \
#       --metadata_dir "metadata/Libri2Mix" \
#       --librimix_outdir "dataset" \
#       --n_src 2 \
#       --freqs 16k \
#       --modes min \
#       --wham_dir "" \
#       --types mix_clean

# --- safety prune ---------------------------------------------------------------
find "$outdir/Libri2Mix/wav16k" -type d \( -name noise -o -name mix_both -o -name mix_raw \) -exec rm -rf {} +

# --- sample-rate sanity ---------------------------------------------------------
find "$outdir/Libri2Mix/wav16k" -name '*.wav' | head -n 10 | xargs soxi -r | grep -qv 16000 && {
  echo "Sample-rate mismatch"; exit 1; }
echo "✅ Clean 16-kHz 2-speaker LibriMix ready in $outdir"

