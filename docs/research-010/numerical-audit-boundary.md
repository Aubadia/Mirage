# Optimizer MR capability and the unchanged numerical-audit boundary

This is a read-only source/routing audit during roadmap010 validation. No backend,
numerical-oracle implementation, numerical threshold or existing certificate rule
was changed for this review. The AST-only check below did not import Torch or
execute either program.

## The concrete tradeoff

The new optimizer MR establishes a previously unavailable conditional comparison:
an actual target Node contribution reaches a registered parameter's accumulated
gradient, that gradient is preserved until the native update boundary, a permitted
representation copy preserves its logical values, and completed source/followup
updates produce comparable parameters and optimizer slots. It does not independently
establish that the original target VJP is mathematically correct. The same incorrect
VJP in both members can yield equal updates and an optimizer-MR Pass.

The [worker](../../src/mirage/runtime/worker.py), `execute_stateful_pair`, still calls
`audit_pair` after selecting and evaluating any MR. Optimizer selection therefore
does **not directly bypass** the auxiliary numerical audit. The selection schedule
is unchanged: explicit enablement, an MR suspect, or every fourth attempt; explicit
disablement takes precedence.

Admission is a separate issue. The [numerical router](../../src/mirage/inspector/oracles.py),
`audit_execution`, requires the old program certificate or the exact
`explicit_straight_line_local_vjp_v1` numerical-source tag. It does not accept the
new `explicit_optimizer_gradient_update_v1` tag. A source containing the newly
reviewed optimizer `step()` consequently remains outside that numerical source
subset. When selected, its audit is `unsupported` with reason
`source effects are outside the existing execution certificate`; when not selected,
it remains `unknown` with the bounded-schedule reason. An optimizer MR Pass does not
change either auxiliary status.

An AST-only counterfactual on [the native MR fixture](../../tests/test_optimizer_relation.py)
confirmed the distinction:

| Source | Program certificate | Local numerical certificate | Optimizer certificate |
| --- | --- | --- | --- |
| Original source, including `ctx['optimizer'].step()` | `None` | `None` | `explicit_optimizer_gradient_update_v1` |
| Same source with only the final `step()` statement removed | `explicit_straight_line_values_v1` | `explicit_straight_line_local_vjp_v1` | `explicit_optimizer_gradient_update_v1` |

The build still creates the same optimizer in the second row; the same target and
backward remain. This demonstrates a source-effect admission difference, **not an
executed numerical agreement or a lost confirmed finding**. Native `Parameter`
construction admitted only by the optimizer certificate can introduce another
source-admission difference. The old certificates deliberately remain unchanged.

## Interpretation and reporting

The genuine capability gain is a verified memory-representation × consumed-gradient
× native-update relation with aligned parameter/slot observations. More such
relations do not establish an overall improvement in defect discovery. Generation
that increasingly emits optimizer histories can simultaneously produce more update
MR comparisons and fewer admitted local numerical audits. Shared oracle code across
treatments does not eliminate this difference in the generated program distribution.

Report the two funnels separately: optimizer-chain validity and MR applicability,
and numerical-audit scheduling, source admission and result. Preserve `mr_verdict`,
`gradient_audits`, their exact abstention reasons and each member's scope. Do not
merge optimizer MR Pass with local VJP agreement, or describe unsupported numerical
histories as numerically validated. This comparison also changes the tested
representation boundary: the optimizer pair changes `.grad` after backward, whereas
an operand-representation pair changes an actual target input. It does not measure
whether backward wrote into the transformed gradient layout.

The post-state screening also retains the existing `dtype_scaled_v1` tolerance:
the backend scales a float64 tensor comparison by its maximum magnitude and
64 machine epsilons. With large parameter magnitudes and tiny updates, different
updates can therefore produce post-parameter differences inside that tolerance.
For momentum0 there is no nonempty momentum slot providing another comparison.
`exact_output_comparison` and `max_abs_parameter_update` are secondary diagnostics;
they do not independently promote a case to suspected failure. This is a
sensitivity boundary of the stated output relation, not evidence that every
different update should be a defect. Future update-sensitive comparisons need
their own numerical error analysis and controls; no threshold was changed after
the final experiment was frozen.

## Possible later extension, not part of the present change

A separate call-local source/effect certificate could admit a captured target/VJP
segment while proving that later supported updates do not affect its replay. It
should identify the exact target call, direct Node, completed backward invocation,
input/output/cotangent snapshots and relevant execution context. The proof must
retain native binding, hook/override, alias and mutation checks, and must account
for effects before the call, between forward and backward, and after capture but
before the audit. A successful native step or a successful replay alone is not such
a proof; globally accepting the optimizer tag would conflate two contracts.

For a first future domain, one could prove that the captured direct backward
completed before a separately verified native update, that the update only changed
the registered parameter/slot state recorded by the observer, and that replay uses
unaltered pre-update snapshots plus certified ambient context. Unsupported external
effects, hidden callbacks, target rebinding or untracked state would remain unknown.
The existing [numerical auditor](../../src/mirage/inspector/numerical_gradients.py)
would still need its read-only eager boundary, supported physical-coordinate chart,
actual cotangents, baseline output/VJP reconstruction, and unchanged difference/
one-sided stability conditions. No new derivative formula or API-specific recipe
is required. Such a certificate needs its own version,
positive/negative controls and Slurm validation; it is not enabled here.
