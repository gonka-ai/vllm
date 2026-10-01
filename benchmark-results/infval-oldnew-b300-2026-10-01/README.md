# Old vs new inference validation — harness

Scripts behind `docs/gonka-infval-old-vs-new-gumbel.md`, which holds the method and results.
DeepSeek-V4-Flash-0731 fp8, 1×B300, image `mlnode:3.1.0-vllm-0.28.0`.

- `harness.py` — the 40 prompts, imported by the other harnesses.
- `harness4.py` — logit-space faces: batched executor pass, batched validator pass and honest
  solo pass in one model load; writes `records4.jsonl`.
- `harness5_exec.py` — executor pass with expert-weight QDQ (`--qbits`, `--qmode int|fp4`,
  `--group`) and logit tuning (`--tau`, `--topp`); writes `commits_<face>.jsonl`.
- `harness5_val.py` — clean-target validator over `commits_*.jsonl` plus the honest solo pass;
  writes merged records.
- `analyze3.py` — scores both validators; edit the face list in the table loop for weight runs.

Run inside the image with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, the weights mounted at `/weights`
and `PYTHONPATH` pointing at this directory. Raw records, commits and logs are git-ignored.
