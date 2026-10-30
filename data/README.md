# Data

This folder is reserved for project datasets and data-related documentation.

## Suggested structure

```text
data/
├── raw/
├── processed/
├── README.md
└── .gitkeep
```

## Guidelines

- Keep raw data separate from processed outputs.
- Do not commit sensitive, restricted, or very large data files unless the licensing allows it.
- Document the source, schema, and cleaning steps used for each dataset.
- Store derived artifacts in `processed/` or another clearly named output folder.

## Notes

The exact dataset for this project should be selected and documented before training the risk model.
