# Local TinySQL export

This directory holds documentation only. Model weights, adapter directories and generated
exports are ignored. From `finetuning/`, after generating data and training local models:

```bash
uv run --project .. --no-sync python -m finetuning.lora_sql.publish \
  --adapter runs/adapter --base runs/base --out runs/exports/tinysql
```

The bundle includes the exact matching base, tokenizer, adapter, inference source,
compatibility information, license and checksums. It requires the documented CPU environment.
Nothing is uploaded automatically, and no public model URL is configured.

Add `--report PATH` to include measured results identifying that adapter's checksum.
Without a matching report, the card claims no execution score. Historical checkpoints
must be restored separately; see the [experiment guide](../README.md).
