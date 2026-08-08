# Test1/Test2 source-matched multi-conformer tasks

These server tasks generate structures only. They do not retrain the model,
select a threshold, or inspect test metrics. Dataset-scoped IDs (`test1__` and
`test2__`) prevent test caches from being confused with training caches.

The generation ranges match the retained training-time conformer branches:

- PepFlow: 3--15 aa; Test1 130 samples and Test2 122 samples.
- AF2/ColabFold: 16--30 aa; Test1 356 samples and Test2 162 samples.
- Every peptide retains 8 conformers. AF2 selects 8 diverse representatives
  from 5 models x 4 seeds after applying the pLDDT quality floor.
- AF2 is deliberately not run for 3--15 aa because that range is assigned to
  PepFlow, and not run for 31--50 aa because the matching training experiment
  was a rejected pilot rather than a retained full-data branch.

Default server paths:

- GHXTox: `/root/autodl-tmp/GHXTox`
- PepFlow: `/root/autodl-tmp/PepFlow`
- PepFlow Python: `/root/autodl-tmp/venvs/pepflow_pip/bin/python`
- LocalColabFold: `/root/autodl-tmp/localcolabfold`

Run one smoke task for each generator, then the full tasks:

```bash
bash scripts/run_test_conformers_autodl.sh pepflow test1 smoke
bash scripts/run_test_conformers_autodl.sh af2 test1 smoke
bash scripts/run_test_conformers_autodl.sh pepflow test1 full
bash scripts/run_test_conformers_autodl.sh pepflow test2 full
bash scripts/run_test_conformers_autodl.sh af2 test1 full
bash scripts/run_test_conformers_autodl.sh af2 test2 full
bash scripts/package_test_conformer_results.sh
```

AF2 can resume from a selected chunk, for example:

```bash
CF_CHUNK_START=3 bash scripts/run_test_conformers_autodl.sh af2 test1 full
```

Chunks with `.msa_complete` and `.predict_complete` markers are skipped.
