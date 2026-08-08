# datasets/

File-based dataset storage. Blobs are **gitignored**; what gets committed is
the layout and the manifests, so the data story stays reproducible.

```
datasets/
├── manifests/    # committed: name, version, path, format, checksum per dataset
├── raw/          # gitignored: data as obtained
└── processed/    # gitignored: training-ready shards
```

Guidelines:

- Synthetic datasets don't live here at all — they are builder functions in
  `mlr.data.synthetic`, registered by name and rebuilt from parameters.
- For real data, drop files under `raw/`, preprocess into `processed/`, and
  add a manifest so `mlr.data` can register a loader for it.
- For large-scale training (e.g. LLM token streams), store memory-mapped
  binary shards or Parquet under `processed/` — never a relational database.
  The SQLite file at the repo root tracks *metadata* (runs, metrics), not
  training data.
