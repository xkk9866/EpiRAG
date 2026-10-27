"""EpiNet: one end-to-end epistemic network for multi-agent RAG over private views.

The package replaces the rule stack of the earlier ``epirag`` prototype
(witness bundle, year executor, relational grammar, hand-set NLI cutoffs and a
hand-written reliability update) with a single trainable model:

* agents answer every plan step from their private pages and send a learned,
  budgeted selection of sentences (``model.comm``);
* the coordinator runs provenance-aware message passing over an epistemic
  graph of steps, claims, sentences, pages and agents, with an evidence
  independence discount and neural truth discovery shared across steps
  (``model.layers``);
* the plan is executed differentiably over the resulting beliefs
  (``model.executor``), so the whole network is trained from final answers.
"""

__version__ = "1.0.0"
