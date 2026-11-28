# Requirement

## Purpose

The component aims to evaluate whether the input tensor matches the target conditions. It will be used to evaluate every freshly generated tensor.
Input a tensor and target conditions, the components will return the evaluation outcome.

## Functional Requirements

### FR-1 — Representation State Inspection

The component must inspect an input tensor and return a structured description of its representation state.

- The description must include the properties supported by this component.
- If a property describe the relationship between tensors, the related tensors must be identified.
- Unsupported inputs must be reported. (Supported scope: torch.strided)

#### Interface

The definition of `RepresentationState` is here: [from typing import Literal

Undetermined = Literal["undetermined"]

class TensorRelationship:
    same_object: bool
    shares_storage: bool | Undetermined
    overlaps_memory: bool | Undetermined

class RepresentationState:
	# Basic metadate
	shape: tuple[int, ...]
	dtype: torch.dtype
	device: torch.device
	layout: torch.layout
	
	# Memory arrangement
	stride: tuple[int, ...]
	storage_offset: int
	storage_nbytes: int
	data_ptr_alignment: int | Undetermined # the largest number of alignment bytes
	is_contiguous: bool
	is_channels_last_contiguous: bool
	is_channels_last_3d_contiguous: bool
	is_pinned: bool
	
	# View and overlap
	is_view: bool
	has_internal_overlap: bool | Undetermined
	is_non_overlapping_and_dense: bool
	
	# Lazy value transformations
	is_conj: bool
  is_neg: bool
	
	# Relationships
	relationships: Mapping[str, TensorRelationship] = field(
        default_factory=dict
    )](https://app.notion.com/p/from-typing-import-Literal-Undetermined-Literal-undetermined-class-TensorRelationship-sa-423fd03e2f728041837ac7e91699b213?pvs=21) 

```python
def inspect_tensor(
    t: torch.Tensor,
    *,
    references: Mapping[str, torch.Tensor] | None = None,
) -> "RepresentationState":
    """Inspect a tensor's representation state.

    Args:
        tensor: The tensor to inspect.
        references: Optional named tensors used to inspect relationships between
                                                        `tensor` and each reference tensor. (Names identify reference
                                                        tensors in the returned state.)

    Returns:
        A RepresentationState containing the supported properties of `tensor`.
        If references are provided, it also contains the supported relationships.
        Properties that cannot be determined are marked as undetermined.

    Raises:
        ValueError: If any input tensor falls outside the component's support scope.
        TypeError: If `tensor` is not a torch.Tensor, or `references` is neither
                                                       None nor a mapping from strings to torch.Tensor objects.
    """
    ...
```

### FR-2 — Target Specification

The component must allow callers to specify the target conditions for supported properties.

- Invalid condition specifications must be rejected with an explanation.

### Interface

```python
class TargetRelationship:
    same_object: bool | None = None
    shares_storage: bool | None = None
    overlaps_memory: bool | None = None

class TargetSpecification:
	# Basic metadate
	shape: tuple[int, ...] | None = None
	dtype: torch.dtype | None = None
	device: torch.device | None = None
	layout: torch.layout | None = None
	
	# Memory arrangement
	stride: tuple[int, ...] | None = None
	storage_offset: int | None = None
	storage_nbytes: int | None = None
	data_ptr_alignment: int | None = None
	is_contiguous: bool | None = None
	is_channels_last_contiguous: bool | None = None
	is_channels_last_3d_contiguous: bool | None = None
	is_pinned: bool | None = None
	
	# View and overlap
	is_view: bool | None = None
	has_internal_overlap: bool | None = None
	is_non_overlapping_and_dense: bool | None = None
	
	# Lazy value transformations
	is_conj: bool | None = None
  is_neg: bool | None = None
	
	# Relationships
	relationships: Mapping[str, TargetRelationship] = field(
        default_factory=dict
    )
```

```python
```python
def validate_target(target: TargetSpecification) -> None:
    """Validate the target specification.

    Args:
        target: The target specification to validate.

    Returns:
		    None if the specification is valild.
        
    Raises:
        ValueError: If a field value violates the documented target rules.
        TypeError: If the target or any field has an invalid type.
    """
    ...
```
```

### FR-3 — Target Satisfaction Evaluation

The component must evaluate whether an actual representation state satisfy the target conditions.

- The outcome could be: satisfied, not satisfied, undetermined.
- For the not satisfied and undetermined outcomes, the result must identify the related conditions and provide an explanation.

```python
from typing import Literal

EvaluationOutcome = Literal[
    "satisfied",
    "not_satisfied",
    "undetermined",
]

class ConditionIssue:
    property_name: str
    reference_name: str | None
    outcome: Literal["not_satisfied", "undetermined"]
    explanation: str
    
class EvaluationResult:
    outcome: EvaluationOutcome
    issues: tuple[ConditionIssue, ...]
```

```python
	def evaluate(
    state: RepresentationState,
    target: TargetSpecification,
) -> EvaluationResult:
    """Evaluate whether a representation state satisfies a target.

    Args:
        state: A valid representation state.
        target: A valid target conditions.

    Returns:
        An EvaluationResult containing the overall outcome and all unsatisfied or 
        undetermined conditions.

    Raises:
        TypeError: If state is not a RepresentationState or target is not a 
						       TargetSpecification.
		    ValueError: If a reference required by target is absent from 
								    state.relationships.
    """
    ...
```

## Constraints & Invariants (Optional)

- INV-1: The input tensor must not be modified during any process.

## Acceptance Criteria

| Requirement ID | Condition | Expected Behavior |
| --- | --- | --- |
| FR-1 | Valid input | Return a `RepresentationState` containing supported properties of the input tensor |
| FR-1 | Invalid argument type | Return a `TypeError` and identify the invalid argument |
| FR-1 | Input tensor out of the supported scope | Return a `ValueError` with explanation |
| FR-1 | Input an empty tensor  | Return a `RepresentationState` without error |
| FR-2 | Valid target condition | Complete the validation and return `None` |
| FR-2 | Invalid condition type | Return `TypeError` and identify the invalid conditions |
| FR-2 | Invalid condition value | Return `ValueError` and explain the violated rule |
| FR-2 | Some target conditions are unspecified | Treat those conditions as unconstrained |
| FR-3 | All conditions are satisfied | Return `satisfied` |
| FR-3 | At least one condition is not satisfied | Return `not_satisfied` and report all mismatched and undetermined conditions with explanations |
| FR-3 | No condition is mismatched but at least one condition is undetermined | Return `Undetermined` and report all undetermined conditions with explanations |
| FR-3 | Invalid argument type | Return a `TypeError` and identify the invalid argument |
| FR-3 | A reference required by the target is absent from the state | Return a `ValueError` and identify the missing reference. |
| INV-1 | After execution | Original input remains unchanged |
