# Data

This folder contains project datasets, reference files, and documentation used for model training, validation, and analysis in the AI Risk Analyzer.

## Suggested structure

```text
data/
├── raw/
│   ├── source_1/
│   └── source_2/
├── processed/
│   ├── features/
│   ├── labels/
│   └── artifacts/
├── docs/
│   └── schema/
├── README.md
├── .gitkeep
└── .gitignore
```

## Guidelines

- Keep raw data separate from processed outputs.
- Do not commit sensitive, restricted, or very large files unless licensing and security requirements permit it.
- Record the source, collection date, schema, and cleaning steps for every dataset.
- Store derived artifacts in `processed/` or another clearly named output folder.
- Use consistent naming conventions and versioned files when data changes materially.
- Add dataset documentation in `docs/` or adjacent markdown files when a schema is non-trivial.
- Prefer reproducible pipelines over ad hoc transformations.

## Notes

The dataset used for risk modeling should be selected, validated, and documented before model training or benchmarking.

Each dataset should include a short description of:

- origin and licensing,
- expected columns / schema,
- preprocessing rules,
- known quality issues,
- train/validation/test split policy when applicable.
