# Mirage Agent Instructions

## Project Scope

Mirage is a research framework for testing deep learning framework correctness across tensor representations. Its three core components have distinct responsibilities.

| Component | Responsibility |
| --- | --- |
| Program Generator | Generate test programs for a target API from its documentation and other available evidence. Generate and validate inputs, verify that the program reaches the target API, and save valid programs and inputs for replay. |
| Representation Mutation Engine | Construct diverse, value-preserving transformation sequences that reach a target representation described by predicates. Preserve the original input, deduplicate paths, and measure transition coverage according to the approved specification. |
| Oracle | Run the same test program on a semantically valid reference input and on the mutated input. Compare the required forward and backward observations, handle execution failures, and record unexplained discrepancies as suspects. |

Read [the architecture specification](specs/architecture.md) and the relevant component specifications before changing component boundaries or shared interfaces. Keep components independently testable and use approved interfaces for cross-component data. Keep search policies, coverage definitions, numerical comparison policies, and service configuration in the relevant specifications.

## Spec-Anchored Development

Spec-Anchored is a form of Spec-Driven Development (SDD) that uses the approved specifications as the shared source of truth for engineers and agents. Before development, agree on requirements, design, and expected behavior. During development, the agent implements the approved specifications while engineers review, make decisions, and verify the work. After development, keep the specifications and code in sync through reviewed changes.

Maintain one pair of specifications per component in both Notion and the repository. Use the same component name in both places. In Notion, the component page can contain both sections. In the repository, keep `requirement.md` and `design.md` together under the component's directory in `specs/`. A pair can cover multiple implementation files.

- `requirement.md` defines the component's purpose, observable behavior, and acceptance criteria. Give functional requirements stable IDs such as `FR-1`. Document public interfaces with type annotations and docstrings, including preconditions, return values, and expected exceptions where applicable. Include relevant constraints, invariants, and an end-to-end example.
- `design.md` defines the technical design and implementation approach. Document the execution flow and relate design decisions to the applicable requirement IDs. Include data structures, algorithms, internal interfaces, and dependencies where they clarify the implementation.
- Humans lead and author the specifications. Agents may review human-authored drafts and flag gaps or inconsistencies. Specification authorship and approval belong to humans.
- Before each implementation task, verify that both corresponding files exist in `specs/` for every affected component, have passed human review, and reflect the approved specifications in Notion. Surface missing or conflicting specifications with the affected behavior and files. Resolve those issues through human review before implementing the affected behavior.

## Development Workflow

1. **Confirm specifications and scope.** Read the approved requirement and design files, shared interfaces, existing implementation, and tests. Identify the affected requirements and state the planned changes, files, and checks before editing. Obtain approval for scope expansion, including changes to shared contracts, dependencies, or repository configuration.
2. **Write unit and end-to-end tests before implementation.** Follow the [Testing](#testing) requirements for coverage, input selection, and expected results.
3. **Have a human review and approve the test baseline.** Freeze the approved baseline before implementation and follow the test change rules below.
4. **Implement against the fixed baseline.** Follow the approved design and iterate on the implementation until all tests pass. Preserve behavior outside the approved scope. Use descriptive names, type annotations on public interfaces, and comments that explain invariants or research-specific decisions.
5. **Refactor and rerun the tests.** Simplify syntax and remove redundant checks while preserving observable behavior and public interfaces. Rerun the suite after refactoring.
6. **Review in a fresh agent session.** Have the reviewer use the approved specifications, tests, and implementation diff to check behavior, coverage, and design compliance. Address findings within the approved scope and rerun the affected checks.
7. **Obtain final human review before merging.** Report the requirements covered, actual test results, skipped or unavailable checks, and unresolved findings. Route specification changes through human review and keep the approved Notion and repository versions synchronized with the code.

## Testing

Write both unit tests and end-to-end (E2E) tests for the affected behavior before implementation. Map each test to the applicable requirement ID and acceptance criterion in the approved `requirement.md`. Derive expected results and failure conditions from those criteria.

- **Unit tests** exercise a component's public behavior in isolation. Check the relevant inputs, outputs, errors, and invariants. For a representation transformation, this can include target predicate attainment, logical value preservation, and preservation of the original input and its required alias relationships.
- **E2E tests** exercise an affected workflow through its public entry point. For a complete Mirage case, verify program and input validation, the observed mutation, and the Oracle's comparison result. Include failure outcomes required by the specification and verify saved records or replay when they are part of the acceptance criteria.
- **Input space partitioning** starts with the dimensions and constraints in the requirements. Identify valid and invalid equivalence classes by expected behavior and select representative inputs from each relevant class. Add edge cases at boundaries and exceptional conditions. For tensors, use relevant dimensions such as shape, dtype, device, layout, and aliasing, and cover combinations when the requirements depend on their interaction.
- **Baseline review** checks the selected cases, expected results, and assertions against the acceptance criteria. Run the new tests before implementation and verify that any expected failures come from missing behavior. Obtain human approval before treating the tests as the fixed baseline.
- **Changes to existing tests** require human review and explicit approval when a test contradicts approved expectations or conflicts with approved new behavior. Otherwise, extend coverage by adding tests. Apply this rule to removals and to changes in assertions, fixtures, tolerances, and configuration that affect existing test behavior.
- **Execution and reporting** use the repository's configured test tools. Run the unit and E2E suites after implementation and again after refactoring. Report the commands, actual outcomes, and any checks skipped or blocked by an unavailable GPU, framework, or LLM endpoint.

## Research Correctness

- Generated programs expose `build` to construct the target API's inputs and `probe` to call the target API. Validate the inputs and use execution evidence to establish reachability, including the relevant backward path when required. Run generated code in an isolated environment with resource limits and without access to credentials or unrelated files.
- Verify logical value preservation, the original input's relevant state and alias relationships, and the target representation predicates at the required execution point. Check the runtime representation after the transformations. Record the ordered path and apply the approved path identity and transition coverage definitions.
- Prefer a contiguous reference input when applicable. Apply the selected metamorphic relation or numerical oracle only when its preconditions, input relation where applicable, and observation requirements hold. Compare corresponding forward and backward observations with the specified tolerances and nondeterminism policy.
- Record invalid or unsupported cases, expected exceptions, crashes, inapplicable comparisons, and suspected correctness failures with their evidence. Reproduce and verify an unexplained discrepancy before reporting a confirmed framework defect.
- Save the program, inputs or reconstruction data, ordered transformations, observed representation states, relevant RNG state and execution settings, framework version, device, dtype, observations, and oracle result as needed for replay. Replay uses saved experiment material without another LLM call. Base coverage and verdicts on recorded evidence.

## Repository and Validation

- Mirage uses a Python `src` layout. Read [`pyproject.toml`](pyproject.toml) and [`.python-version`](.python-version) for the environment. The configured CLI resolves `mirage:main` through `src/mirage/__init__.py`.
- Keep generated experiment outputs and credentials out of source changes.
