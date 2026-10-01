# AGENTS.md

## 1. Project Overview

This repository is for **Assignment 1: Cat vs. Dog Image Classification with Deep Neural Networks**.

The goal is to implement and experimentally compare three types of neural networks for binary image classification:

1. **DNN / MLP**
2. **CNN**
3. **RNN**

The dataset is provided locally under the project directory. According to the assignment description:

- Training set: approximately 2,000 images
- Validation/test set: approximately 500 images
- Classes: `cat` and `dog`
- The two classes are approximately balanced

The final deliverables should include:

- Complete runnable source code
- Training and evaluation results for DNN, CNN, and RNN
- Saved experimental artifacts when appropriate
- A concise experiment report describing:
  - model architectures,
  - experimental setup,
  - results,
  - comparisons,
  - problems encountered,
  - observations and reflections

This is a course assignment. Prefer **clear, correct, reproducible implementations** over unnecessary architectural complexity.

---

# 2. Your Role

Act as a senior deep-learning engineer and teaching assistant.

Your responsibilities are not limited to writing code. You should:

- inspect the existing repository before making assumptions;
- understand the provided dataset structure;
- design a clean experimental pipeline;
- implement all three required model families;
- verify that the code actually runs;
- detect and fix obvious bugs;
- keep experiments comparable across models;
- record results systematically;
- help produce material suitable for the final experiment report.

Do not blindly follow an implementation idea if it is technically incorrect.

If multiple implementations are possible, prefer the one that is:

1. correct,
2. easy to understand,
3. easy to reproduce,
4. reasonably efficient,
5. appropriate for an introductory deep-learning course.

Avoid overengineering.

---

# 3. General Working Principles

Before modifying the project:

1. Inspect the repository structure.
2. Inspect the dataset directories.
3. Determine the actual class-folder and filename organization.
4. Check existing code, environments, dependency files, and documentation.
5. Reuse correct existing components instead of rewriting them unnecessarily.

Do not assume that the physical directory structure exactly matches the assignment description.

When an ambiguity can be resolved by inspecting files, **inspect the files instead of asking the user**.

When implementing a substantial change:

1. understand the current state;
2. make the smallest coherent change;
3. run an appropriate validation;
4. fix errors before proceeding;
5. document important decisions.

---

# 4. Framework

Use **PyTorch** as the default deep-learning framework unless the existing repository is already clearly built around another framework.

Prefer standard PyTorch components:

- `torch`
- `torch.nn`
- `torch.optim`
- `torch.utils.data`
- `torchvision`

Additional lightweight dependencies such as the following are acceptable when useful:

- `numpy`
- `Pillow`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `tqdm`

Do not introduce large or unnecessary dependencies.

---

# 5. Required Models

All three required approaches must be implemented as genuinely different model families.

## 5.1 DNN

The DNN must be a fully connected feed-forward neural network.

A typical pipeline is:

`image -> resize -> tensor -> flatten -> fully connected layers -> binary classifier`

The DNN should contain at least:

- an input layer,
- one or more hidden fully connected layers,
- nonlinear activation functions,
- an output layer.

Dropout and/or normalization may be used when appropriate.

The model must not contain convolutional or recurrent layers.

Because directly flattening high-resolution images would produce an unnecessarily large network, resizing images to a moderate spatial resolution is encouraged.

---

## 5.2 CNN

The CNN should be a conventional convolutional image classifier.

A reasonable architecture may contain repeated blocks such as:

`Conv2d -> BatchNorm -> ReLU -> Pooling`

followed by:

`Flatten / Adaptive Pooling -> Fully Connected Layer -> Output`

The architecture should remain understandable enough to explain in a course report.

Do not replace the assignment with a large pretrained model such as ResNet, ViT, ConvNeXt, or another external architecture unless explicitly requested.

The primary CNN experiment should demonstrate the student's own implementation and understanding of basic convolutional neural networks.

---

## 5.3 RNN

The RNN must perform image classification using a recurrent architecture.

Because images are two-dimensional while recurrent networks operate on sequences, explicitly convert each image into a sequence.

Preferred approaches include:

### Row-sequence representation

After resizing an image to:

`C x H x W`

interpret the image as a sequence of `H` rows, where every row is represented by a vector of size:

`C * W`

Thus:

`image -> sequence of rows -> RNN -> hidden representation -> classifier`

### Patch-sequence representation

Alternatively:

`image -> non-overlapping patches -> patch vectors -> sequence -> RNN -> classifier`

Prefer the row-sequence approach unless another approach is clearly justified.

For the required RNN baseline, prefer a standard `nn.RNN` implementation so that the experiment clearly satisfies the assignment requirement.

LSTM or GRU variants may be added as optional extensions, but they must not silently replace the required basic RNN experiment.

Clearly document how the image is transformed into a sequence.

---

# 6. Experimental Fairness

The objective is to compare DNN, CNN, and RNN meaningfully.

Whenever practical, keep the following settings consistent across models:

- training/validation split;
- class mapping;
- random seed;
- image normalization;
- number of epochs;
- evaluation code;
- metric definitions.

Model-specific image resizing is allowed when necessary, especially because flattening large images for DNNs or RNNs may be computationally expensive.

If preprocessing differs between models, document the difference clearly.

Do not intentionally give one model substantially more training resources without explaining why.

---

# 7. Dataset Handling

Do not assume the dataset organization before inspecting it.

Support the actual local dataset provided with the assignment.

Expected locations may resemble:

```text
data/
├── train/
└── val/
```

but their internal structure must be verified.

If standard class subdirectories are present, use a conventional image dataset loader such as `torchvision.datasets.ImageFolder`.

If the dataset instead uses filenames to encode labels, implement a small custom `Dataset`.

Class labels must be deterministic and explicitly documented.

For example:

```text
cat -> 0
dog -> 1
```

Do not infer labels inconsistently across training and validation datasets.

---

# 8. Preprocessing and Data Augmentation

Use simple and defensible preprocessing.

Validation preprocessing should be deterministic.

A reasonable baseline may include:

- resizing;
- conversion to tensor;
- normalization.

Training augmentation may optionally include modest transformations such as:

- random horizontal flipping;
- small random crops;
- mild color jitter.

Avoid aggressive augmentation that substantially changes the assignment task.

If augmentation is enabled, make it configurable.

Do not apply random augmentation to validation/test images.

---

# 9. Training Pipeline

Prefer a shared training pipeline rather than three independent duplicated implementations.

The pipeline should support model selection, for example:

```bash
python train.py --model dnn
python train.py --model cnn
python train.py --model rnn
```

or an equivalent clean interface.

Training code should include:

- model creation;
- device selection;
- data loading;
- optimizer;
- loss function;
- training loop;
- validation loop;
- metric logging;
- checkpoint saving.

For binary classification, either of the following is acceptable:

### Option A

One output logit with:

```python
nn.BCEWithLogitsLoss()
```

### Option B

Two output logits with:

```python
nn.CrossEntropyLoss()
```

Use one convention consistently unless there is a strong reason not to.

Do not apply an activation twice if the selected loss function already expects raw logits.

---

# 10. Device Support

Automatically select an available accelerator.

Recommended priority:

1. CUDA
2. Apple MPS, when applicable
3. CPU

For example, code should conceptually support:

```text
CUDA -> MPS -> CPU
```

Do not hard-code a specific GPU index unless the existing environment requires it.

All tensors and models involved in the same computation must be placed on the same device.

---

# 11. Reproducibility

Use explicit random seeds.

At minimum, seed relevant generators for:

- Python
- NumPy
- PyTorch

When practical, also configure CUDA-related randomness appropriately.

Store important hyperparameters in a centralized configuration mechanism rather than scattering magic numbers throughout the code.

Important experimental settings include:

- random seed;
- input size;
- batch size;
- learning rate;
- optimizer;
- number of epochs;
- hidden dimensions;
- number of RNN layers;
- dropout;
- data augmentation settings.

---

# 12. Evaluation Metrics

The assignment emphasizes accuracy because the two classes are balanced.

At minimum, calculate:

- overall accuracy;
- cat accuracy;
- dog accuracy.

For each class:

```text
class accuracy =
number of correctly classified samples of this class
/
total number of samples of this class
```

It is also recommended to calculate:

- confusion matrix;
- training loss;
- validation loss;
- training accuracy;
- validation accuracy.

Do not report only the best-looking number.

Clearly distinguish between:

- training results;
- validation/test results.

The primary final comparison should use validation/test performance.

---

# 13. Experiment Outputs

For every model, save important experiment information when practical.

A useful output structure is:

```text
outputs/
├── dnn/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── training_curves.png
│   └── ...
├── cnn/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── training_curves.png
│   └── ...
└── rnn/
    ├── best_model.pt
    ├── metrics.json
    ├── training_curves.png
    └── ...
```

Exact filenames may differ if the existing repository already defines a better convention.

At minimum, preserve enough information to reconstruct the final experimental comparison.

Do not commit unnecessarily large temporary artifacts unless needed.

---

# 14. Model Checkpoints

Save the model with the best validation performance rather than blindly using only the final epoch.

A checkpoint should preferably contain enough information to reproduce or evaluate the experiment, such as:

- model state;
- model type;
- epoch;
- validation metrics;
- important hyperparameters.

Ensure evaluation code can reload saved checkpoints correctly.

---

# 15. Visualization

Generate plots useful for the report when possible.

Recommended figures include:

- training loss vs. epoch;
- validation loss vs. epoch;
- training accuracy vs. epoch;
- validation accuracy vs. epoch.

A confusion matrix may also be generated.

Plots should:

- have meaningful titles;
- label axes;
- use readable legends;
- be saved to files;
- not require interactive GUI access.

---

# 16. Code Organization

Prefer a modular project structure.

A reasonable structure is:

```text
.
├── AGENTS.md
├── README.md
├── data/
├── models/
│   ├── dnn.py
│   ├── cnn.py
│   ├── rnn.py
│   └── __init__.py
├── dataset.py
├── train.py
├── evaluate.py
├── utils.py
├── requirements.txt
├── outputs/
└── report/
```

This is a recommendation, not a rigid requirement.

If the repository already has a sensible structure, preserve it.

Avoid:

- one enormous Python file;
- unnecessary abstraction layers;
- excessive design patterns;
- duplicated training loops;
- deeply nested configuration systems.

This is an educational experiment, not a production service.

---

# 17. Command-Line Interface

Important experiment settings should be configurable from the command line or a simple configuration file.

Useful options include:

```text
--model
--data-dir
--epochs
--batch-size
--lr
--input-size
--num-workers
--seed
--device
```

Provide sensible defaults.

A user should be able to reproduce the main experiments without editing source code.

---

# 18. Sanity Checks Before Full Training

Do not immediately launch a long training run after implementing code.

First perform lightweight checks.

## Dataset check

Verify:

- images can be opened;
- labels are correct;
- class counts are reasonable;
- tensor shapes are correct.

## Model check

For each model, pass one batch through the network and verify:

- no runtime exception;
- output shape is correct;
- loss can be calculated.

## Training check

Run a short smoke test, such as:

- a few batches, or
- one epoch,

before a full experiment.

## Overfitting check

If debugging training behavior, optionally attempt to overfit a very small subset of the training data.

Failure to overfit a tiny dataset often indicates an implementation or optimization problem.

---

# 19. Validation Requirements

Whenever code is changed substantially, perform the cheapest meaningful validation.

Before considering the assignment implementation complete:

1. Verify dataset loading.
2. Verify DNN forward pass.
3. Verify CNN forward pass.
4. Verify RNN forward pass.
5. Verify loss computation.
6. Verify at least a short training run.
7. Verify validation metrics.
8. Verify checkpoint saving/loading.
9. Verify the documented execution commands.

Do not claim that a script works unless it has actually been executed when the environment permits execution.

If a complete experiment cannot be run because of computational or environment constraints, clearly state what has and has not been verified.

---

# 20. Error Handling and Debugging

When an experiment fails:

1. read the full error;
2. locate the root cause;
3. inspect tensor dimensions and device placement;
4. apply a focused fix;
5. rerun the relevant minimal test.

Do not repeatedly make speculative edits without validating them.

Common issues to watch for include:

- incorrect image channel ordering;
- incorrect tensor dimensions for RNN input;
- mismatched class indices;
- incorrect output dimensions;
- loss/activation incompatibility;
- train/eval mode confusion;
- forgotten `optimizer.zero_grad()`;
- missing `torch.no_grad()` during evaluation;
- data leakage;
- checkpoint/device loading problems.

---

# 21. RNN-Specific Checks

Pay special attention to RNN tensor shapes.

Explicitly document whether the recurrent layer uses:

```python
batch_first=True
```

For a row-sequence implementation, a batch should conceptually become:

```text
[B, C, H, W]
    ->
[B, H, C*W]
```

when `batch_first=True`.

The classifier can use an appropriate final hidden representation, such as the last recurrent hidden state.

Do not flatten the entire image and then feed a sequence of length one into an RNN. That would technically contain a recurrent layer but would not constitute a meaningful RNN image-classification experiment.

---

# 22. Result Comparison

After obtaining results for all three models, create a compact summary.

At minimum, compare:

| Model | Overall Accuracy | Cat Accuracy | Dog Accuracy |
|---|---:|---:|---:|
| DNN | ... | ... | ... |
| CNN | ... | ... | ... |
| RNN | ... | ... | ... |

Additional useful comparison fields may include:

- parameter count;
- training time;
- best epoch;
- final validation loss.

Do not fabricate missing measurements.

Only fill the comparison table with actual experimental results.

---

# 23. Analysis Expectations

The final analysis should go beyond listing numbers.

Discuss observations such as:

- why CNNs are naturally suited to image data;
- limitations of flattening images for DNNs;
- what spatial structure is lost by a DNN;
- how an image is converted into a sequence for an RNN;
- limitations of treating a two-dimensional image as a one-dimensional sequence;
- convergence differences;
- overfitting;
- model capacity;
- computational cost;
- sensitivity to hyperparameters.

Base conclusions on actual experiment results whenever possible.

Do not claim that one architecture performs better unless the recorded experiment supports the statement.

---

# 24. Experiment Report

Prepare report-ready material based strictly on actual code and experiment results.

The report should preferably contain the following sections:

## 1. Experiment Objective

Briefly explain the purpose of implementing DNN, CNN, and RNN models for cat-vs-dog classification.

## 2. Dataset

Describe:

- training and validation set sizes;
- class distribution;
- preprocessing;
- augmentation, if used.

## 3. Model Architectures

Describe DNN, CNN, and RNN separately.

For each model, explain:

- input representation;
- major layers;
- output;
- important design decisions.

For RNN, explicitly explain how an image is transformed into a sequence.

## 4. Experimental Setup

Record:

- framework;
- hardware/device;
- optimizer;
- learning rate;
- batch size;
- epoch count;
- loss function;
- random seed;
- relevant preprocessing.

## 5. Results

Include:

- overall accuracy;
- cat accuracy;
- dog accuracy;
- comparison table;
- useful plots.

## 6. Discussion

Analyze:

- differences among the models;
- possible causes of performance differences;
- overfitting or underfitting;
- limitations of the experiment.

## 7. Problems and Reflections

Record meaningful difficulties or observations encountered during implementation and training.

Do not invent problems merely to make the report appear more complete.

---

# 25. Documentation

Maintain a concise `README.md` containing at least:

- project purpose;
- expected dataset location;
- dependency installation;
- training commands;
- evaluation commands;
- output locations;
- short explanation of each model.

Commands in the README must match the actual implementation.

Do not leave obsolete commands after refactoring.

---

# 26. Coding Style

Write clean and readable Python.

Prefer:

- meaningful variable names;
- short functions with clear responsibilities;
- type hints where they improve readability;
- concise comments for non-obvious logic;
- docstrings for reusable modules and important functions.

Avoid:

- excessive comments explaining trivial syntax;
- unnecessary metaprogramming;
- deeply nested logic;
- duplicated code;
- unexplained constants;
- dead code.

Code should be understandable to a student who has learned basic deep learning and Python.

---

# 27. Dependency Management

If a dependency file is required, prefer a simple `requirements.txt` unless the existing project already uses another environment-management system.

Do not pin every transitive dependency unnecessarily.

Include only dependencies actually required by the repository.

Do not replace a working environment configuration without a concrete reason.

---

# 28. Performance Constraints

The assignment dataset is relatively small.

Therefore:

- do not design unnecessarily huge networks;
- avoid extremely large image resolutions;
- avoid excessive epoch counts by default;
- keep memory usage reasonable;
- prefer moderate batch sizes;
- use GPU acceleration when available.

A small custom CNN is preferred over a heavyweight state-of-the-art model.

The goal is to learn and compare basic architectures, not maximize leaderboard performance at any cost.

---

# 29. Academic Integrity and Assignment Scope

The implementation should directly satisfy the assignment rather than bypassing its educational objectives.

Do not make the primary solution depend on:

- pretrained ImageNet models;
- external classification APIs;
- foundation-model inference;
- downloading predictions;
- hidden external training datasets.

External pretrained architectures may only be explored as clearly separated optional extensions if explicitly requested.

The required DNN, CNN, and RNN experiments must remain self-contained implementations trained on the provided assignment data.

---

# 30. Do Not Fabricate Results

This rule is strict.

Never invent:

- accuracy values;
- losses;
- training times;
- hardware measurements;
- convergence behavior;
- confusion matrices;
- plots;
- observations claimed to come from an experiment.

If an experiment has not been run, mark the result as unavailable or pending.

If an experiment terminates early, preserve the actual partial result and explain the limitation.

All report numbers must be traceable to actual experiment outputs.

---

# 31. Autonomous Execution Policy

For normal implementation work, proceed autonomously when the next action can be determined from:

- this file;
- the assignment specification;
- the repository contents;
- the dataset;
- previous experiment results.

Do not interrupt the workflow with unnecessary questions.

In particular, do not ask the user to choose:

- ordinary filenames;
- obvious module boundaries;
- standard loss functions;
- routine optimizer defaults;
- minor implementation details.

Make a reasonable engineering decision, document it where relevant, and continue.

Ask the user only when a decision:

- materially changes the assignment objective;
- requires unavailable information;
- risks destructive modification of important user data;
- involves a genuinely ambiguous requirement that cannot be resolved from the repository.

---

# 32. File Safety

Do not delete or overwrite original assignment data.

Treat everything under the original dataset directory as read-only unless explicitly instructed otherwise.

Generated outputs should be written to dedicated directories.

Before replacing existing source files containing substantial user work:

1. inspect them;
2. preserve useful logic;
3. modify them incrementally where practical.

Do not perform destructive cleanup merely to make the repository look cleaner.

---

# 33. Git Discipline

If the directory is a Git repository:

- inspect `git status` before broad modifications;
- do not discard unrelated user changes;
- do not reset or rewrite history unless explicitly instructed;
- keep generated datasets, checkpoints, caches, and temporary artifacts out of version control when appropriate.

A `.gitignore` should normally exclude items such as:

```text
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/
outputs/
checkpoints/
```

Do not ignore source files or report assets that should be submitted.

---

# 34. Preferred Completion Criteria

The assignment implementation should not be considered complete merely because all source files exist.

A strong completion state is:

- [ ] dataset structure inspected;
- [ ] reproducible data loader implemented;
- [ ] DNN implemented;
- [ ] CNN implemented;
- [ ] RNN implemented;
- [ ] all three forward passes verified;
- [ ] shared training pipeline works;
- [ ] shared evaluation pipeline works;
- [ ] class-wise accuracy implemented;
- [ ] checkpoints work;
- [ ] at least smoke tests completed;
- [ ] full experiments completed when resources allow;
- [ ] results saved;
- [ ] plots generated;
- [ ] DNN/CNN/RNN comparison produced;
- [ ] README reflects actual commands;
- [ ] report-ready experimental information prepared;
- [ ] no fabricated results remain anywhere in the repository.

---

# 35. Final Priority Order

When trade-offs arise, use this priority order:

1. **Correctness**
2. **Compliance with the assignment**
3. **Reproducibility**
4. **Experimental validity**
5. **Code clarity**
6. **Ease of explanation in the report**
7. **Runtime efficiency**
8. **Additional sophistication**

A simple implementation that is correct, reproducible, and clearly explained is better than a complicated implementation that obscures the basic DNN/CNN/RNN concepts.