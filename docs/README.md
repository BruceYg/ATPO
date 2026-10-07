# Documentation

**Using the models**

- [inference.md](inference.md): `atpo predict` and `VideoSafetyModel`; choosing a
  checkpoint; inference settings and presets; the output format.
- [evaluation.md](evaluation.md): `atpo evaluate`; how incomplete predictions are
  counted; metric definitions; the paper's evaluation sets.

**Training**

- [custom_training.md](custom_training.md): ATPO on your own categories, end to end.
- [paper_setup.md](paper_setup.md): the paper's pipeline on SafeWatch and
  XD-Violence.
- [data_preparation.md](data_preparation.md): canonical records, validation,
  recipes, and the paper's datasets.
- [training.md](training.md): SFT, GRPO, and ATPO launches; run directories;
  export; hardware.
- [controller.md](controller.md): the reward, the controller update, its settings,
  timing, validation, and checkpoint/resume semantics.
- [correctness_fixes.md](correctness_fixes.md): where release defaults differ from
  the paper's code, and how to restore the original behaviour.

Elsewhere in the repository:

- [environments/README.md](../environments/README.md);
- [prompts/README.md](../prompts/README.md);
- [third_party/EasyR1/PATCHES.md](../third_party/EasyR1/PATCHES.md).
