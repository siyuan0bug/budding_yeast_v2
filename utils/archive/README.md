# Historical active learning

`al_callback_legacy.py` is the complete, unchanged pre-archival callback snapshot
from 2026-10-21. It preserves `random/us/is/wrs/vessal/hggs/pial`, their original
initialization, helpers and global dependencies. It also includes the contemporary
online methods to make the snapshot self-contained. Production code does not import
this module; old strategy names are rejected by the current training CLI.

Explicit reproduction import:

```python
from budding_yeast_v2.utils.archive.al_callback_legacy import ActiveLearningCallback
```

Use the historical training driver/configuration for full experiment reproduction;
importing this class alone does not restore an old training run or its state.
See `docs/AL_ARCHIVAL_AND_OFFICIAL_ALIGNMENT_2026-10-21.md` for the migration audit.

`al_callback_before_package.py` preserves the exact 2931-line implementation
immediately before package extraction. `bmdal_compat.py` in this archive resolves
its historical relative imports. See `docs/AL_PACKAGE_REFACTOR.md`.

`biological_guard.py` preserves the exact pre-cleanup 121-line helper module.
Its projection functions had no production callers; historical helper tests import
this archive explicitly. Production `utils/biological_guard.py` retains unchanged
variable groups only. Archiving does not validate the old projection assumptions.
