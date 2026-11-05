# Planner — Unsloth training, Transformers validation

Notebook: [LifeMate Two Node Planner Qwen35](https://www.kaggle.com/code/cymeu4l0t/lifemate-two-node-planner-qwen35).

Select **GPU T4 ×2**, enable Internet, and attach the private competition dataset. The standalone notebook embeds source and the fixed split of 200 validation IDs; it does not embed competition data. It is saved for manual execution, not launched automatically.

The Kaggle connector did not persist the requested dataset attachment. Before running, use **Add Input** to attach your private competition dataset (currently cymeu4l0t/zaloai-dataset). The notebook detects either its extracted train folder or a ZIP. Publishing status is recorded in published.json; Kaggle's save status does not establish that training has run.

Set PROFILE="smoke" for four optimizer updates on 16 training and eight validation episodes. Set PROFILE="full" for two epochs and every reference-prefix decision from the 200 validation episodes. Use a new RUN_NAME each time. Full starts from the pretrained model, never a smoke adapter.

Both modes use microbatch2/GPU × two DDP workers × accumulation2 = effective batch8. Thinking is disabled through the chat template. Only completion JSON is supervised. No input truncation, packing, synthetic recovery examples, revise, output repair or token-level enforcer.

Qwen3.5/T4 training preserves Unsloth's forced-FP32 computation path when requested by the library, while loading the unquantized base with FP16 storage. It does not force FP16 autocast onto Gated DeltaNet. Precision and memory are recorded for each rank. The exact combined DDP workload still requires a GPU smoke run; CPU tests are not a GPU compatibility or performance claim.

There is no validation during training. The final adapter is exported, training processes exit, and two fresh ordinary Transformers/PEFT processes generate decisions on disjoint validation prefixes. Paired base evaluation disables the adapter on the identical frozen backbone. Metrics are computed from raw JSON output and include every invalid answer. They measure reference-prefix planning, not complete agent success or the official contest score.

The input schema, messages, completion encoding, training process and generation metrics are separated under src/twonode/. Caller can be added as its own independently supervised task later; it is not implemented here.

Rebuild after changing runtime source:

```bash
python notebooks/planner/make_notebook.py
```

The requirements pin the Qwen3.5/T4 dependency family. packages.txt captures the actual resolved environment. config.json records the model revision, and source_manifest.json records the embedded source hashes.

Keep notebook/output private. artifacts.zip contains the final adapter, prepared examples, manifests and reports; intermediate checkpoints and compile caches remain outside that archive. No training, GPU smoke or validation-quality run is performed by publishing the notebook.

Startup diagnostics record PID/rank, stage duration, process RSS, available system RAM and allocated/reserved GPU memory before/after import, model loading, LoRA attachment and Trainer initialization. If a stage takes longer than 120 seconds, faulthandler repeatedly records all Python thread stacks in diagnostics/rankN.STAGE.stacks.log. The timer is diagnostic, not a termination deadline. These files remain under the output directory even when startup fails; no optimizer/model changes are implied by this diagnostic update.
