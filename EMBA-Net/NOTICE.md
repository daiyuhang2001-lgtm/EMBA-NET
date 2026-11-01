# Source and modification notice

This package is a focused extraction of the user-supplied `ultralytics-main.rar`,
whose `ultralytics/__init__.py` declares version 8.2.50. That archive is a modified
Ultralytics YOLO codebase. This package is not represented as an independently
invented detection framework, and its folder/application name does not change
the provenance of its operators or architecture.

The original Ultralytics AGPL-3.0 license is retained in `LICENSE`. The upstream
project location recorded in the supplied pyproject.toml is
https://github.com/ultralytics/ultralytics/ . The provided fork also contains
third-party/custom modules; their individual authorship is not established by
the archive alone. Original module/class names are recorded in the rename map,
with explanatory references and source-location records retained. No claim of ownership over these prior
operators is made. Additional source-specific notices should be preserved if
the author supplies them later.

See `docs/source_map.json` for the source file, source line span and source-file
hash associated with each extracted definition.

Changes made during this extraction:

- Retained only `EMBA-Net.yaml`, the six selected top-level module types and
  their active dependencies.
- The YAML now uses paper-facing module names; its original `nc: 80`, graph
  connections, arguments and scale definitions are unchanged. Examples explicitly
  select `nc=1` and scale `n`.
- Kept the original active `C2f_EMA` implementation, its local Conv/Bottleneck,
  the SBA update order, and shared DEConv/GN head. No RepConv was inserted.
- Replaced three CPU/default-CUDA tensor allocations inside DEConv kernel
  assembly with `weight.new_zeros`, preserving the computation while respecting
  the weight's dtype and device.
- Added a restricted builder that follows the original parser's channel/scale
  and repeat rules for these modules, with strict state-dict loading.
- Reduced the loss to the active TAL/BCE/CIoU/DFL path. The original loss.py had
  a leading `CLOU` token and imported many unused alternatives; neither is needed
  in this focused implementation.
- Added small preprocessing, NMS, training, prediction and optional ONNX tools.
  These convenience tools are new examples; they do not reproduce the complete
  historical Ultralytics trainer, augmentation, optimizer grouping or schedule.
- For fixed-shape ONNX export only, replaced EMA adaptive directional pooling
  with mathematically equivalent means and explicit reshape dimensions. Normal
  training and inference still use the original EMA forward definition.
- At the user's request, renamed SBA to BGSA, the complete C2f_EMA block to
  GLM_P_B (paper display name "GLM-P B"), and Detect_LSDECD to
  LSDheadreplacement. The internal EMA attention helper is named GLMContext;
  its member prefix changed from `.EMA.` to `.context.` with strict legacy-key
  migration in load_weights. The previous module names remain documented as
  provenance and compatibility information, not as a claim of new operators.
- Removed unrelated model definitions, alternative experiments, other task
  frameworks, datasets, docs/examples, plots, IDE files and caches from the
  deliverable. The user's original archive and desktop checkout were untouched.

Verification compares extracted original definitions and the original
`parse_model` function in an isolated namespace. It does not claim successful
import or execution of every file in the supplied full fork. No trained weights
were included in the supplied archive. Random/synthetic smoke-test weights are
not included in this deliverable and do not establish the paper's accuracy.

## PVFeaturePerceptionNet backbone packaging in version 0.1.2

The existing ten Conv/C2f/SPPF stages are grouped under the application-facing
name PVFeaturePerceptionNet. The main YAML invokes this named backbone and the focused
builder expands its original stage specification. The registered model now has
backbone/head containers; load_weights migrates the old flat tensor-key prefixes
before strict loading. Original stage operations, features, graph connections,
parameter tensor shapes and provenance are retained. The name is a grouping
label, not a claim that the underlying backbone/operators are newly invented.

## Runnable dataset integration in version 0.2.0

The model configuration is named EMBA-Net.yaml and its default class count is
set to one for the supplied bird-dropping dataset. The previous examples also
explicitly used one class, so their model parameter count and architecture are
unchanged. All current entry-point references use the new configuration name.
The user-supplied data2 dataset is bundled under data/ while preserving its
existing split and weather augmentations at the user's explicit request.
Only deterministic file-pair repairs and quarantining of unresolved missing
labels were performed, as recorded in data_review. Convenience launchers and
run.py were added; no historical experiment results or pretrained weights
are claimed or included.


## Paper-aligned structural revision, version 0.3.0 (2026-10-29)

The default graph now uses glm_paper.GLM_P_B and lsd_paper.LSDheadreplacement.
The original definitions remain available for the original implementation configuration.
New RepConv follows established multi-branch reparameterization, with two active
3x3 branches plus a 1x1 branch in the selected budget configuration. This is not a
new-operator claim. The GLM uses an equal EMA/local split and RepConv before and
after the split. The LSD implements scale-weighted P3 latent alignment, shared
DEConv, residual scale decoding and separate DW/PW classification/regression branches.
The source backbone, BGSA, detection scales and loss remain based on the supplied code.
Per-stage hidden widths and head width were chosen using operation counts only;
no validation or test accuracy was used in budget selection.
The new model has not completed research training and does not inherit the paper's
historical detection or deployment metrics. Full details and counting exclusions
are in README.md and verification.json.


## Figure-topology revision v0.4.0 (supersedes the v0.3 structural draft)

GLM now follows the Figure 2 EMA-before-split path and concatenates the bypass
and all serial bottleneck outputs, using ReLU RepConv. The Figure 4 projection
implements the 2C split, DS convolution and three multiplicative gates. Its
middle gate is C/2, explicitly resolving the figure's incompatible C/4 label.
LSD runs its shared Conv/DEConv/Conv on each native scale; the prior common P3
latent and cross-scale residuals are removed. BGSA follows equations (1)-(5)
with jointly generated gates and simultaneous updates. Figure/text conflicts,
normalization choices and actual strides are documented in STRUCTURE_MAP.md.
Channel widths and active RepConv branches were selected only using parameter
and operation budgets. All research performance requires new training.


## Budget refinement v0.4.1

The v0.4 topology is retained. GLM hidden widths are 24/4/8/4/36 and active
3x3 branch counts are 1/1/1/2/2. This yields 2,313,702 training-graph parameters
and 8.72752256G deployment convolution/GEMM operations at 1x3x640x640.
Configuration selection uses operation and parameter counts only; no accuracy
result or old-model performance is claimed for the revised model.
