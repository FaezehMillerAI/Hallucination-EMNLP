
## Full proposal

**TITLE PROJECT**  
**Claim-Grounded Causal Hallucination Detection and Mitigation in Medical Vision-Language Models via Cross-Modal Self-Consistency** 
**OBJECTIVE**  
Implement a unified framework that detects, localizes, explains, and mitigates hallucinations in medical Vision-Language Model outputs without using any knowledge graph, ontology, UMLS, RadLex, or SNOMED-CT resource. The method should operate at the atomic-claim level and use only image evidence, token dynamics, hidden states, cross-modal attention, counterfactual perturbations, and causal interventions to identify unsupported claims and correct them during decoding. 

**BACKGROUND CONTEXT – Problem Statement**  
The attached draft targets hallucinations in medical VLMs and highlights three central gaps: graph-based hallucination methods focus on text-only models, uncertainty-based methods lack structured grounding, and neuron-level probes detect errors but do not close the control gap through mitigation. A KG-free redesign should retain the same detect-localize-explain-mitigate scope while replacing symbolic grounding with stronger claim-level evidence verification grounded directly in model behavior and image support. 
**BACKGROUND CONTEXT – Key Innovation**  
This framework introduces a claim-grounded causal evidence paradigm in which every generated claim is tested for visual support, representation stability, and intervention sensitivity. Instead of aligning claim tokens to knowledge graph paths as in the attached design, the method builds an internal cross-modal evidence graph from claims, text tokens, visual patches, and decoder states, then performs causal reasoning over support and conflict patterns to detect hallucinations and guide correction. 

**ARCHITECTURE OVERVIEW**  
The framework consists of 5 modules:  
1. Atomic Claim Decomposition  
2. Cross-Modal Evidence Graph Extraction  
3. Causal Hallucination Detection  
4. Intervention-Guided Attention Reallocation for Mitigation  
5. Explainability Layer 
### MODULE 1 Atomic Claim Decomposition

**1.1 Goal**  
Decompose each VLM-generated answer or report into atomic medical claims that can be evaluated independently for factual support. This mirrors the attachment’s claim decomposition step but removes KG entity mapping and instead preserves claim spans, semantic role structure, and uncertainty cues for direct evidence checking. 

**1.2 Input Format**  
Each sample consists of:  
- Medical image, DICOM or PNG  
- Question for VQA or generation prompt for report generation  
- VLM-generated response  
- Optional ground-truth answer or reference report for supervision and pseudo-labeling 

**1.3 Claim Extraction Procedure**  
For each generated response:  
1. Split into sentences.  
2. Extract atomic claims using biomedical NER, dependency parsing, and relation pattern templates.  
3. Represent each claim as a tuple: subject, predicate, object, modifiers, negation, uncertainty, anatomical location, severity, token span indices.  
4. Preserve a claim-to-token alignment map for downstream evidence attribution. 

**1.4 Claim Types**  
The parser should identify at least these categories:  
- Finding presence or absence, such as consolidation present  
- Anatomical localization, such as opacity in the left lower lobe  
- Severity or extent, such as mild cardiomegaly  
- Comparison or progression, such as increased pleural effusion  
- Normality statements, such as no acute cardiopulmonary abnormality 

**1.5 Output**  
- `claim_decompositions/{sample_id}.json`  
- Fields: sentence text, atomic claims, token spans, claim type, assertion polarity, uncertainty markers, anatomical phrase spans 

### MODULE 2 Cross-Modal Evidence Graph Extraction

**2.1 Supported Models**  
Primary target medical VLMs can follow the same evaluation spirit as the attached draft, including LLaVA-Med, Med-Flamingo, and RadFM, with general VLMs such as LLaVA-1.5 and Qwen-VL as comparison models. The attachment explicitly frames these as the core model families for analysis and cross-model evaluation. 

**2.2 Attention and State Extraction**  
For each image-text pair and VLM:  
1. Run a forward pass with `output_attentions=True` and `output_hidden_states=True`.  
2. Extract decoder token logits, decoder hidden states, cross-attention maps, and vision encoder patch embeddings.  
3. Store layerwise and headwise trajectories rather than only mean attention.  
4. Compute token confidence, entropy, and logit margin for each generated token. 

**2.3 Heterogeneous Evidence Graph**  
Construct an attributed graph with four node types:  
- `claim`: one node per atomic claim  
- `texttoken`: generated token nodes  
- `visualpatch`: image patch nodes  
- `latentstate`: compressed layerwise decoder state nodes

**2.4 Node Features**  
- `claim`: pooled token embeddings, claim type, uncertainty indicators, negation markers  
- `texttoken`: token embedding, logit entropy, logit margin, positional index, hidden state trajectory  
- `visualpatch`: patch embedding, spatial coordinates, saliency score, masked-response sensitivity  
- `latentstate`: per-layer decoder summary, residual norm, attention concentration 
**2.5 Edge Types**  
- `claim-to-token`: claim span membership  
- `token-to-token`: self-attention dependency  
- `token-to-patch`: cross-modal attention  
- `patch-to-patch`: visual self-attention or spatial adjacency  
- `token-to-latentstate`: hidden-state provenance  
- `claim-to-patch`: attribution-based support edge  
- `claim-to-claim`: contradiction, redundancy, or consistency relation within the same output 

**2.6 Edge Attributes**  
- Attention weight  
- Attribution strength  
- Counterfactual sensitivity score  
- Semantic similarity  
- Temporal stability across layers  
- Cross-head variance 

**2.7 Counterfactual Probing**  
For each claim, generate perturbed evidence profiles using:  
- Patch masking of top attended visual regions  
- Random control masking of non-salient regions  
- Local blur or intensity perturbation  
- Token dropout in claim spans  
- Decoder layer ablation or head suppression for high-contribution heads 

**2.8 Output**  
- `evidence_graphs/{sample_id}.pt` as PyTorch Geometric `HeteroData`  
- `counterfactual_profiles/{sample_id}.json` containing claim-level perturbation responses and sensitivity statistics 

### MODULE 3 Causal Hallucination Detection

**3.1 Model Architecture**  
Implement a Heterogeneous Graph Transformer or equivalent relational graph model over the evidence graph. The attached draft uses an HGT over text, visual, and KG nodes; here the same graph-learning spirit is preserved but all reasoning occurs over claim, token, patch, and latent-state nodes only. 

**3.2 Core Principle**  
A claim is likely hallucinated when it shows one or more of the following patterns:  
- Weak or diffuse grounding to image patches  
- High confidence but low visual support  
- Instability under small evidence-preserving perturbations  
- Strong dependence on decoder priors rather than image-conditioned evidence  
- Internal contradiction with other claims in the same output 

**3.3 Multi-Task Prediction Heads**  
For each atomic claim, predict:  
- Binary hallucination label  
- Cause label  
- Evidence sufficiency score  
- Causal faithfulness score  
- Localization quality score 

**3.4 Cause Taxonomy**  
Use four classes:  
- `no_hallucination`  
- `visual_misinterpretation`  
- `prior_driven_fabrication`  
- `context_misalignment`

**3.5 Novel Scoring Functions**  
Define the following claim-level signals:

- **Visual Support Score**: average attribution mass from claim tokens to causally important visual patches.  
- **Trajectory Stability Score**: consistency of token-to-patch alignment across layers and heads.  
- **Counterfactual Consistency Score**: degree to which a claim changes only when supporting visual evidence is removed.  
- **Decoder Prior Dominance Score**: confidence retained after image evidence degradation.  
- **Intra-Response Consistency Score**: compatibility of a claim with neighboring claims in the same answer or report. 

**3.6 Causal Faithfulness Definition**  
For claim \(c_i\), let \(p_i\) be the original claim probability and \(p_i^{(-S)}\) be the claim probability after removing supportive patch set \(S\). Define causal faithfulness as:  
\[
F_i = \frac{p_i - p_i^{(-S)}}{p_i + \epsilon}
\]
A supported claim should show a meaningful drop under removal of evidence, whereas a fabricated claim may remain largely unchanged because it is driven by language prior rather than visual evidence. This replaces the attachment’s KG-path versus attention-path alignment score with an intervention-based support measure. 

**3.7 Training Strategy**  
Train with supervised labels where available and pseudo-labels otherwise. The attached draft already proposes pseudo-label generation through atomic-fact alignment and cause taxonomy assignment, so this KG-free version keeps that philosophy but bases pseudo-cause labels on visual support failure, perturbation sensitivity, and context mismatch instead of KG contradiction checks. 

### MODULE 4 Intervention-Guided Attention Reallocation for Mitigation

**4.1 Goal**  
Correct hallucinated claims by modifying decoding behavior so the model relies more heavily on causally supportive visual evidence and less on unsupported prior-driven token trajectories. This is the direct KG-free counterpart to the attachment’s graph-guided attention reallocation module. 

**4.2 Mitigation Procedure**  
For each detected hallucinated claim:  
1. Identify unsupported claim tokens and the corresponding token-to-patch edges with low evidence sufficiency.  
2. Detect high-prior heads and layers using prior dominance and perturbation insensitivity.  
3. Downweight unsupported cross-attention edges.  
4. Upweight edges toward patches with strong causal contribution and stable support trajectories.  
5. Re-run generation from the earliest hallucination onset token. 
**4.3 Causal Reallocation Rule**  
For token \(t\) and patch \(v\), let \(A_{tv}\) be attention, \(E_{tv}\) evidence strength, and \(D_t\) prior dominance. A simple reweighting rule is:  
\[
A'_{tv} = \text{Normalize}\left(A_{tv} \cdot (1 + \alpha E_{tv} - \beta D_t)\right)
\]
where \(\alpha\) controls support amplification and \(\beta\) penalizes prior-dominated unsupported generation. This parallels the attachment’s multiplicative attention update but does not rely on any KG-supported region or symbolic path. 

**4.4 Decoding Modes**  
Implement two mitigation variants:  
- **Inference-time correction** using attention hooks and local re-decoding  
- **Training-time correction** by fine-tuning with an evidence-aligned decoding objective 

**4.5 Output**  
- `mitigation/corrected_outputs/{sample_id}.json`  
- `mitigation/attention_edits/{sample_id}.json`  
- `mitigation/intervention_traces/{sample_id}.json`

### MODULE 5 Explainability Layer

**5.1 Claim Explanation Graph**  
For each hallucinated claim, extract the minimal subgraph connecting claim nodes, token nodes, key visual patches, and influential latent states. The attachment’s explainability layer outputs subgraphs connecting hallucinated claims, attended visual regions, and KG reasoning paths; the KG-free version keeps the same explanation format but substitutes intervention traces and support/conflict edges for symbolic KG paths.

**5.2 Visual Explanations**  
Generate heatmaps showing:  
- Attended regions  
- Causally supportive regions  
- Unsupported but over-attended regions  
- Before-versus-after mitigation differences 
**5.3 Textual Explanations**  
For each claim, produce a JSON explanation with:  
- Claim text  
- Hallucination flag  
- Predicted cause  
- Evidence sufficiency score  
- Faithfulness score  
- Top supportive patches  
- Top conflicting patches  
- Perturbation effects  
- Decoder heads implicated in fabrication 

**5.4 Example Explanation Format**  
```json
{
  "claim": "The chest X-ray shows cardiomegaly.",
  "hallucination_detected": true,
  "cause": "prior_driven_fabrication",
  "evidence_sufficiency": 0.21,
  "causal_faithfulness": 0.14,
  "supportive_regions": [
    {"patch_id": 67, "spatial_location": "cardiomediastinal silhouette", "evidence_score": 0.32}
  ],
  "unsupported_regions": [
    {"patch_id": 45, "spatial_location": "left upper lung", "attention_weight": 0.78}
  ],
  "counterfactual_effect": {
    "mask_supportive_patch_drop": 0.06,
    "mask_unsupported_patch_drop": 0.01
  },
  "conflict": "Claim confidence remains high despite removal of image evidence, indicating decoder-prior dominance."
}
```  
This preserves the attached draft’s explanation-oriented output style while removing the KG reasoning path field entirely. 

## Training losses

Use a multi-task loss:

\[
L_{\text{total}} = L_{\text{hall}} + \lambda_1 L_{\text{cause}} + \lambda_2 L_{\text{evidence}} + \lambda_3 L_{\text{faith}} + \lambda_4 L_{\text{local}} + \lambda_5 L_{\text{mitigation}}
\]

where:  
- \(L_{\text{hall}}\): binary cross-entropy for hallucination detection  
- \(L_{\text{cause}}\): cross-entropy for hallucination cause classification  
- \(L_{\text{evidence}}\): mean squared error or ranking loss for evidence sufficiency  
- \(L_{\text{faith}}\): regression loss for causal faithfulness  
- \(L_{\text{local}}\): localization loss against segmentation masks or region annotations when available  
- \(L_{\text{mitigation}}\): KL-divergence or sequence-level consistency loss between original and corrected decoding distributions 

Recommended starting weights:  
- \(\lambda_1 = 0.5\)  
- \(\lambda_2 = 0.3\)  
- \(\lambda_3 = 0.3\)  
- \(\lambda_4 = 0.2\)  
- \(\lambda_5 = 0.4\) 

For pseudo-labeling on datasets without explicit hallucination labels, use atomic fact decomposition plus consistency scoring against reference answers, then assign causes by heuristics: low support and high perturbation sensitivity suggests visual misinterpretation, low support and low perturbation sensitivity suggests prior-driven fabrication, and high support but question-inconsistent answers suggests context misalignment. This follows the attached draft’s pseudo-labeling philosophy while replacing KG contradiction tests with intervention-based diagnostics. 

## Evaluation

The attached draft already identifies MedHEval, HEAL-MedVQA, Med-HallMark, MedVH, and MedHallTune as relevant resources, and HEAL-MedVQA specifically provides chest X-ray VQA with anatomical region masks that are useful for localization and visual grounding evaluation. MedHEval also includes evaluation code and baseline organization, which fits well for reproducing claim-level hallucination and mitigation benchmarking. 

### DATASETS – Primary Evaluation Datasets

1. **MedHEval**  
- Use for claim-level hallucination detection, cause classification, and mitigation benchmarking across medical VLM outputs.  
- The repository provides evaluation and baseline code organization for these experiments.

2. **HEAL-MedVQA**  
- Use for grounding and localization evaluation because it includes chest X-ray medical VQA with anatomical regions and mask annotations.  
- The dataset page reports 19,231 total samples, with train and test splits and region masks suitable for supervision of evidence localization. 

3. **Med-HallMark / MedVH / MedHallTune**  
- Use for broader hallucination stress testing, cross-task evaluation, and generalization studies, consistent with the benchmark scope listed in the attached draft. 

### EVALUATION METRICS – Detection

- AUROC at claim level and sentence level  
- F1, precision, recall for binary hallucination detection  
- Macro-F1 across hallucination causes  
- Calibration error for hallucination confidence 

### EVALUATION METRICS – Mitigation

- Hallucination rate reduction before versus after correction  
- Factual accuracy improvement after mitigation  
- ALFA-style atomic fact consistency improvement  
- Regeneration quality preservation, such as BLEU, ROUGE-L, or clinical factual overlap against references 

### EVALUATION METRICS – Explainability

- Localization IoU or pointing accuracy against region masks where available  
- Evidence sufficiency correlation with human judgments  
- Deletion and insertion curves for supportive patches  
- Human expert ratings for explanation clarity and clinical usefulness, consistent with the human-evaluation direction in the attached draft. 

### EVALUATION METRICS – Clinical Safety

- False negative rate for hallucination detection  
- Hallucination severity breakdown by claim type  
- Cross-model generalization, for example train on LLaVA-Med and test on Med-Flamingo or RadFM, following the cross-model setup proposed in the attachment. 

### BASELINES TO IMPLEMENT

- UniVRSE-style uncertainty baseline  
- CHARM-style graph baseline adapted without KG  
- VCD decoding baseline  
- OPERA decoding baseline  
- Direct linear probing baseline  
- Attention-only ablation without counterfactual intervention  
- Causal-only ablation without graph learning 
### EXPERIMENTAL SETUP

- Hardware and training style can follow the attached draft’s practical scope, which assumes multi-GPU high-memory training and about 48 hours for the full original pipeline. The KG-free version should be computationally lighter in preprocessing because it removes ontology parsing and KG embedding construction, though counterfactual probing may still be expensive. 

### CODE STRUCTURE

```text
project_root/
  config/
    model_config.yaml
    training_config.yaml
    dataset_config.yaml
    intervention_config.yaml

  src/
    claim_parser/
      sentence_splitter.py
      biomedical_ner.py
      claim_decomposer.py
      uncertainty_parser.py

    evidence_extraction/
      vlm_wrapper.py
      attention_extractor.py
      hidden_state_extractor.py
      attribution_maps.py
      perturbation_engine.py
      evidence_graph_builder.py

    detector/
      hgt_model.py
      multitask_heads.py
      faithfulness.py
      evidence_scores.py
      cause_classifier.py

    mitigation/
      attention_hooks.py
      causal_reallocation.py
      regenerate.py

    explainability/
      explanation_graph.py
      heatmaps.py
      intervention_trace.py
      report_generator.py

    baselines/
      univrse.py
      vcd.py
      opera.py
      linear_probe.py
      attention_only.py

    evaluation/
      metrics.py
      localization_eval.py
      mitigation_eval.py
      clinical_safety.py
      human_eval.py

  data/
    raw/
    processed/

  results/
    detection/
    mitigation/
    explanations/
    figures/
```
This mirrors the attached draft’s code-organization style while removing every KG-specific directory and replacing it with perturbation, evidence, and intervention modules. 

## AI agent prompt

Below is a rewritten implementation prompt you can give directly to an AI agent.

***

**PROMPT FOR AI AGENT**

Design and implement a full research framework titled:

**Claim-Grounded Causal Hallucination Detection and Mitigation in Medical Vision-Language Models via Cross-Modal Self-Consistency**

Your task is to produce a complete technical proposal and implementation-ready system design for a medical VLM hallucination framework that is fully **knowledge-graph-free**.

### Core requirements

Build a unified framework that can:

1. Detect hallucinations in medical VLM outputs.  
2. Localize evidence for each claim on the medical image.  
3. Explain why a claim is hallucinated.  
4. Mitigate or correct hallucinated claims during decoding.  

### Strict constraints

- Do **not** use any knowledge graph, ontology, symbolic graph, UMLS, RadLex, SNOMED-CT, RDF, OWL, or external medical concept graph.
- Do **not** rely on entity-to-KG linking, KG path search, KG embeddings, or KG-guided correction.
- The framework must be fully based on:
  - atomic claim decomposition,
  - cross-modal attention,
  - hidden states,
  - token logits and entropy,
  - visual patch embeddings,
  - attribution maps,
  - counterfactual perturbations,
  - causal intervention,
  - self-consistency analysis.

### Required scientific contribution

The method must have strong publication-level novelty. Include at least these novelties:

1. Atomic claim-level hallucination modeling.  
2. Cross-modal evidence graph with no KG nodes.  
3. Layerwise attention trajectory analysis.  
4. Counterfactual visual consistency scoring.  
5. Causal intervention-based decoding correction.  

### Required output structure

Write the proposal in this exact format:

1. TITLE PROJECT  
2. OBJECTIVE  
3. BACKGROUND CONTEXT  
   - Problem Statement  
   - Key Innovation  
4. ARCHITECTURE OVERVIEW  
5. MODULE 1 Atomic Claim Decomposition  
6. MODULE 2 Cross-Modal Evidence Graph Extraction  
7. MODULE 3 Causal Hallucination Detection  
8. MODULE 4 Intervention-Guided Attention Reallocation for Mitigation  
9. MODULE 5 Explainability Layer  
10. TRAINING LOSSES  
11. DATASETS  
12. BASELINE METHODS TO IMPLEMENT  
13. EVALUATION METRICS  
14. EXPERIMENTAL SETUP  
15. CODE STRUCTURE  

### Detailed implementation instructions

#### MODULE 1
Implement atomic claim decomposition for generated medical answers or reports.

For each response:
- split into sentences,
- extract atomic claims,
- identify subject, predicate, object, anatomical location, negation, uncertainty, severity,
- preserve token-span alignment.

Output:
- claim decomposition JSON per sample.

#### MODULE 2
Build a heterogeneous cross-modal evidence graph from:
- claim nodes,
- text token nodes,
- visual patch nodes,
- latent decoder state nodes.

Extract from the VLM:
- cross-attentions,
- hidden states,
- patch embeddings,
- token probabilities,
- token entropy,
- headwise and layerwise trajectories.

Add edge types such as:
- claim-to-token,
- token-to-token,
- token-to-patch,
- patch-to-patch,
- token-to-latentstate,
- claim-to-patch,
- claim-to-claim.

Compute edge attributes such as:
- attention weight,
- attribution score,
- semantic similarity,
- perturbation sensitivity,
- temporal stability across layers,
- head variance.

Also implement counterfactual probing:
- patch masking,
- random masking control,
- local blur,
- token dropout,
- attention-head suppression,
- decoder-layer ablation.

Output:
- PyTorch Geometric `HeteroData` graph per sample,
- counterfactual profile JSON per sample.

#### MODULE 3
Implement a causal hallucination detector using a Heterogeneous Graph Transformer or equivalent graph model.

For each atomic claim, predict:
- binary hallucination label,
- cause label,
- evidence sufficiency score,
- causal faithfulness score,
- localization quality score.

Use this cause taxonomy:
- no_hallucination,
- visual_misinterpretation,
- prior_driven_fabrication,
- context_misalignment.

Define and compute:
- Visual Support Score,
- Trajectory Stability Score,
- Counterfactual Consistency Score,
- Decoder Prior Dominance Score,
- Intra-Response Consistency Score.

Define causal faithfulness using intervention on supportive patch sets and claim probability drop.

#### MODULE 4
Implement mitigation by intervention-guided attention reallocation.

For each hallucinated claim:
- identify unsupported claim tokens,
- identify low-evidence token-to-patch edges,
- identify prior-dominated heads and layers,
- downweight unsupported attention edges,
- upweight causally supportive visual regions,
- re-run decoding from hallucination onset.

Support both:
- inference-time correction with hooks,
- training-time correction with evidence-aligned fine-tuning.

Save:
- corrected outputs,
- attention edits,
- intervention traces.

#### MODULE 5
Implement explainability outputs at claim level.

Produce:
- explanation subgraph,
- heatmaps for attended vs supportive vs unsupported regions,
- textual JSON explanation for each claim,
- before/after mitigation comparisons.

### Training instructions

Use a multi-task objective including:
- hallucination detection loss,
- cause classification loss,
- evidence sufficiency regression loss,
- causal faithfulness regression loss,
- localization loss,
- mitigation loss.

Provide equations and suggested coefficient weights.

Also include pseudo-label generation for datasets without claim-level hallucination labels by using:
- atomic fact comparison with references,
- visual support heuristics,
- perturbation sensitivity heuristics,
- context consistency heuristics.

### Datasets

Use these as primary candidates:
- MedHEval for hallucination benchmarking and mitigation evaluation,
- HEAL-MedVQA for localization and region-grounded evaluation,
- Med-HallMark,
- MedVH,
- MedHallTune.

When describing HEAL-MedVQA, note that it includes chest X-ray VQA with anatomical regions and masks, and the dataset page reports 19,231 total samples. 

When describing MedHEval, note that the repository includes evaluation code, baseline organization, and data generation structure. 

### Baselines

Include these baseline families:
- uncertainty-based baseline,
- graph baseline without KG,
- VCD,
- OPERA,
- linear probe,
- attention-only ablation,
- causal-only ablation.

### Evaluation

Include metrics for:
- hallucination detection,
- cause classification,
- mitigation quality,
- factual improvement,
- localization quality,
- explanation quality,
- clinical safety,
- cross-model generalization.

### Code structure

Provide a full directory tree for:
- claim parsing,
- evidence extraction,
- detector,
- mitigation,
- explainability,
- evaluation,
- baselines,
- configs,
- results.

### Final output requirements

Your final response must be:
- technically detailed,
- implementation oriented,
- written in formal research style,
- structured exactly like a proposal,
- free of any knowledge-graph components,
- strong enough to guide actual coding by a research engineer.

***

## Best refinement

One important improvement over the attached draft is that this version is not merely “the same system minus KG”; instead, it replaces KG consistency with intervention-based causal faithfulness, claim-wise evidence sufficiency, and trajectory stability, which makes the novelty much stronger and easier to defend as a stand-alone paper contribution. The attached draft’s original novelty rests on combining KG grounding with cross-modal graph reasoning, so this rewrite needs the causal consistency machinery to remain comparably strong. 

