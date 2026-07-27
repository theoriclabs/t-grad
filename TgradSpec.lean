import Tgrad.Ontology
import Tgrad.Spec.Epistemic
import Tgrad.Spec.Architecture
import Tgrad.Spec.Findings
import Tgrad.Spec.LiveConditions
import Tgrad.Spec.RuntimeWork
import Tgrad.Spec.Growth
import Tgrad.Spec.Evolution
import Tgrad.Spec.Work

/-! # TgradSpec — checked specification, separate from the product runtime

`Tgrad` is the product library. `TgradSpec` is the epistemic and operational
model used to inspect and plan changes to it. Keeping separate roots ensures
the spec is compiled and tested without linking roadmap/finding data into the
Python-facing shared library.
-/
