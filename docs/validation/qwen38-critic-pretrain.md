# Qwen3.8 critic pretraining validation

Date: 2026-08-28

This report records the implementation and validation boundary for the
Slime-native offline critic pretraining workflow. It does not claim that a
27B checkpoint has been trained.

## Local Mac verification

The focused CPU/static suite passed:

```text
25 passed in 3.42s
```

It covered the deterministic Orchard artifact helpers, token/mask normalization,
trajectory-equal loss, DP batch packaging, prior-centered scalar-head
initialization, checkpoint selection, the offline launcher, and the online
role-specific critic-load path. `ruff check examples/coding_agent_rl/critic_pretrain`
also passed.

The broader role-config test was not green in the Mac environment because its
existing dependency stack is incomplete (`wandb` and `ray` are not installed).
That is an environment limitation, not evidence about the new trainer.

## Development-machine boundary

The known workflow tunnel was healthy and the target checkout was inspected:

```text
workflow=workflow_62220389
remote_root=/personal/muchen/slime-sao
branch=codex/sao...origin/codex/rollpacker [ahead 13, behind 1]
```

The remote checkout already contains unrelated dirty SAO changes. Uploading the
new source files and tests was therefore prepared as an explicit allowlist, but
the desktop approval boundary rejected private source-file egress. No new
critic-pretrain file was uploaded, and no remote training or validation result
is claimed here.

## Pending remote gates

After explicit authorization for the allowlisted sync, run the following on the
development machine:

1. prepare the pinned `microsoft/Orchard/swe` artifact;
2. run the 4,096-trajectory full-parameter canary;
3. reload the selected native checkpoint and verify no value-head
   reinitialization warning;
4. evaluate the isolated test split;
5. run the paired online random-head versus pretrained-critic SAO canary with
   `sao-critic-warmup-steps=4`.

The remote artifact, checkpoint paths, metrics, and online first-ten-update
comparison must be appended here only after those commands produce real output.
