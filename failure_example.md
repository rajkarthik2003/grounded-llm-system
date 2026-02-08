### Example: Hallucination Prevention

**Question**
What are common risks of hypertension?

**Baseline Behavior**
The model generated a confident list of risks (heart disease, stroke, kidney failure)
even though the retrieved documents did not contain this information.

**Issue**
The answer appeared fluent but was unsupported by the source context.

**After Grounding + Refusal Logic**
The system refused to answer:
"I cannot answer safely. The provided context does not mention common risks of hypertension."

**Outcome**
The system prevented an unsupported medical claim and prioritized correctness.
