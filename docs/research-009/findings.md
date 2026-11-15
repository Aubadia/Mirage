# Mirage research findings

## Current understanding

Phase-008 negative results show that a larger measurable goal vocabulary is not
sufficient for efficient testing. The useful path is goal proposal -> selection ->
legal execution -> runtime attainment -> applicable correctness evidence. Cost and
failure at each boundary matter. Initial causal explanations remain hypotheses.

## Lessons and constraints

- Do not conflate an available goal, an LLM plan, runtime attainment and MR Pass.
- Keep coverage-only exploration and API breadth; favoring easy MR passes alone
  would undermine Mirage's memory/training-state contribution.
- Do not use issue-specific inputs, target-VJP formula templates in generation,
  or relaxed validity/numerical gates to improve the reported score.
- Freeze protocols and snapshots before execution. Prior ten benchmark APIs and
  mul/cos smoke APIs have been seen; reserve fresh APIs for a generality check.
- Preserve original reports, the user's index and Slurm-only computation rules.

## Open questions

How much regression comes from proposal/scheduling bias, redundant repairs,
operand-binding mistakes, unavailable legal transitions, and certificate support?
Which improvements transfer to previously unexamined APIs and preserve novel
tensor-state reachability while increasing useful tests per unit cost?

## Preformal results

The first development trial was negative on validity/MR throughput; retained in full. Fixed-request encoding support met its preregistered static-contract adoption criterion (3/4 differential recoveries,0/3regressions), without executing those generated programs. The second development trial supports a descriptive improvement (valid19→25,MR12→18,qualified joint3→26) but does not isolate individual component causality. Explicit memory×training conjunction remains absent. Final16API×2arm campaign is frozen and running once withseed0; no tuning after formal observations.

Final static certificate counterfactual on445prior sources admits19previously uncertified originally-valid sources and removes0. No runtime or MR reclassification follows from this static result. Shared certificate/resolver corrections occur inboth comparison arms.

## Formal synthesis and postformal correction

Formal comparison is mixed: valid110→105,MR70→78,qualifiedjoint17→54allfresh; plannedvalid29→17,memorygoalvalid19→2,realbackwardlifecycle2→2,explicitmemory×training0. This doesnotvalidateimproved plannedinteractionsearch. Contextlimit failures and residual redundant schema obligations further limitplanning.

A posthoc savedcase audit establishes a moreconcrete missedbug mechanism: candidate slogdet8 reachednegativeviewandobservedbackward butshared whole-source certificate blockedbothMRandnumeric audit duelegalrunrequires_grad_. IndependentFractionandminimal ordinary/negative/materialized controls corroborate localknownbackwardissue. Formalautonomoussuspectsremain0. A separate restrictedlocalVJPsourceeffectcertificate hasbeenimplemented; numericalalgorithm/MRcertificate unchanged. Its validation andsamecase replay remain distinctfrom formalmetrics.
