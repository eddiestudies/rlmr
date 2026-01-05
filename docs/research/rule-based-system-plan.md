# Research Plan: Self-Designing Rule-Based System for RLM

## Executive Summary

This research extends the RLM (Recursive Language Model) paradigm to enable LLMs to **design, maintain, and invoke their own rule-based systems**. Rather than always deferring to LLM calls for sub-tasks, the system learns to create programmatic rules that can be executed deterministically when the LLM is confident they will succeed.

### Research Questions

1. **Rule Emergence**: Can LLMs effectively design rule-based systems for recurring patterns they encounter?
2. **Confidence Calibration**: How well can LLMs predict when rules will succeed vs. when LLM reasoning is needed?
3. **Performance Trade-offs**: What are the latency, cost, and accuracy implications of rule-based shortcuts?
4. **Interpretability Gains**: Do rules provide meaningful interpretability into the LLM's decision-making?
5. **Self-Improvement**: Do rules improve in quality over iterative updates by the LLM?

---

## 1. Architecture Overview

### 1.1 Current RLM Flow
```
User Query → RLM → REPL Environment → llm_query() → Sub-LLM → Response
                                    ↓
                              Execute Code
                                    ↓
                              FINAL Answer
```

### 1.2 Proposed Rule-Augmented Flow
```
User Query → RLM → REPL Environment → rule_query() → RuleRouter
                                                         ↓
                                    ┌────────────────────┴────────────────────┐
                                    ↓                                         ↓
                              [High Confidence]                        [Low Confidence]
                                    ↓                                         ↓
                              RuleExecutor                              llm_query()
                                    ↓                                         ↓
                              Rule Result ←──── Feedback Loop ←───── LLM Response
                                    ↓                                         ↓
                              (+ Rule Update Proposal)                  (+ Rule Creation Proposal)
```

### 1.3 Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **RuleStore** | Persistent storage for rules with versioning | `rlm/rules/store.py` |
| **RuleExecutor** | Execute rules and validate outputs | `rlm/rules/executor.py` |
| **RuleRouter** | Decide: rule vs LLM based on confidence | `rlm/rules/router.py` |
| **RuleLogger** | Track rule usage, performance, evolution | `rlm/rules/logger.py` |
| **RulePrompts** | Prompts for rule creation/update by LLM | `rlm/rules/prompts.py` |

---

## 2. Rule System Design

### 2.1 Rule Definition Schema

```python
@dataclass
class Rule:
    """A rule created by the LLM for deterministic execution."""

    # Identity
    id: str                          # Unique identifier (uuid)
    name: str                        # Human-readable name
    version: int                     # Version number (for evolution tracking)

    # Pattern Matching
    trigger_pattern: str             # Regex or semantic pattern for when to apply
    input_schema: dict               # JSON schema for expected input
    preconditions: list[str]         # Python expressions that must be True

    # Execution
    rule_type: Literal["python", "template", "lookup", "transform"]
    code: str                        # Python code, template, or lookup table
    output_schema: dict              # JSON schema for expected output

    # Confidence & Routing
    confidence_threshold: float      # Min confidence to use this rule (0-1)
    fallback_to_llm: bool           # Whether to fallback on rule failure

    # Metadata
    created_by_model: str           # Which LLM created this rule
    created_at: datetime
    updated_at: datetime
    description: str                # LLM's explanation of what this rule does

    # Performance Tracking
    total_invocations: int
    successful_invocations: int
    failed_invocations: int
    avg_execution_time_ms: float

    # Lineage
    parent_rule_id: str | None      # If this is an evolution of another rule
    creation_context: str           # The query/task that prompted rule creation
```

### 2.2 Rule Types

| Type | Description | Example Use Case |
|------|-------------|------------------|
| **Python** | Executable Python function | Data transformations, calculations |
| **Template** | String template with placeholders | Structured output formatting |
| **Lookup** | Key-value mapping table | Entity resolution, category mapping |
| **Transform** | Regex/pattern-based text transforms | Parsing, extraction, normalization |

### 2.3 Example Rules

```python
# Example 1: Python rule for date parsing
Rule(
    name="parse_date_formats",
    trigger_pattern=r"parse.*date|extract.*date|date.*format",
    rule_type="python",
    code="""
from dateutil import parser
def execute(input_text: str) -> dict:
    try:
        parsed = parser.parse(input_text, fuzzy=True)
        return {"date": parsed.isoformat(), "success": True}
    except:
        return {"success": False, "fallback": True}
""",
    confidence_threshold=0.85,
    fallback_to_llm=True
)

# Example 2: Lookup rule for common abbreviations
Rule(
    name="state_abbreviation_lookup",
    trigger_pattern=r"state.*abbreviation|abbrev.*state",
    rule_type="lookup",
    code='{"CA": "California", "NY": "New York", ...}',
    confidence_threshold=0.95,
    fallback_to_llm=False  # Deterministic lookup
)

# Example 3: Template rule for structured output
Rule(
    name="format_citation",
    trigger_pattern=r"format.*citation|cite.*format",
    rule_type="template",
    code="{author} ({year}). {title}. {journal}, {volume}({issue}), {pages}.",
    confidence_threshold=0.90,
    fallback_to_llm=True
)
```

---

## 3. Component Specifications

### 3.1 RuleStore (`rlm/rules/store.py`)

```python
class RuleStore:
    """Persistent storage for rules with versioning and search."""

    def __init__(self, storage_path: str):
        """Initialize with path to rule storage (JSON/SQLite)."""

    # CRUD Operations
    def create_rule(self, rule: Rule) -> str:
        """Create a new rule, return rule ID."""

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a rule by ID."""

    def update_rule(self, rule_id: str, updates: dict) -> Rule:
        """Update a rule, incrementing version."""

    def delete_rule(self, rule_id: str) -> bool:
        """Soft-delete a rule (maintain history)."""

    # Search & Matching
    def find_matching_rules(self, query: str, context: dict) -> list[RuleMatch]:
        """Find rules whose trigger_pattern matches the query."""

    def get_rules_by_type(self, rule_type: str) -> list[Rule]:
        """Get all rules of a specific type."""

    # Evolution & History
    def get_rule_history(self, rule_id: str) -> list[Rule]:
        """Get all versions of a rule."""

    def get_rule_lineage(self, rule_id: str) -> list[Rule]:
        """Get parent chain of evolved rules."""

    # Analytics
    def get_rule_stats(self) -> dict:
        """Get aggregate statistics on rule usage."""
```

### 3.2 RuleExecutor (`rlm/rules/executor.py`)

```python
class RuleExecutor:
    """Safe execution environment for rules."""

    def __init__(self, timeout_ms: int = 1000):
        """Initialize with execution timeout."""

    def execute(self, rule: Rule, input_data: dict) -> RuleResult:
        """
        Execute a rule safely.

        Returns:
            RuleResult with:
                - success: bool
                - output: Any (the result)
                - execution_time_ms: float
                - error: str | None
                - should_fallback: bool
        """

    def validate_input(self, rule: Rule, input_data: dict) -> bool:
        """Validate input against rule's input_schema."""

    def validate_output(self, rule: Rule, output: Any) -> bool:
        """Validate output against rule's output_schema."""

    def check_preconditions(self, rule: Rule, context: dict) -> bool:
        """Check if all preconditions are satisfied."""
```

### 3.3 RuleRouter (`rlm/rules/router.py`)

```python
class RuleRouter:
    """Decides whether to use a rule or fallback to LLM."""

    def __init__(
        self,
        rule_store: RuleStore,
        confidence_model: str = "heuristic",  # or "learned"
        min_confidence: float = 0.7
    ):
        """Initialize router with rule store and confidence settings."""

    def route(self, query: str, context: dict) -> RoutingDecision:
        """
        Decide how to handle a query.

        Returns:
            RoutingDecision with:
                - use_rule: bool
                - rule: Rule | None
                - confidence: float
                - reasoning: str
        """

    def compute_confidence(self, rule: Rule, query: str, context: dict) -> float:
        """
        Compute confidence that rule will succeed.

        Factors:
            - Pattern match strength
            - Rule's historical success rate
            - Input schema match
            - Precondition satisfaction
            - Query similarity to rule's creation context
        """

    def should_create_rule(self, query: str, llm_response: str) -> bool:
        """
        After LLM response, should we propose creating a rule?

        Heuristics:
            - Query is similar to past queries
            - Response follows a consistent pattern
            - Task is deterministic/well-defined
        """
```

### 3.4 RuleLogger (`rlm/rules/logger.py`)

```python
@dataclass
class RuleInvocation:
    """Record of a single rule invocation."""
    rule_id: str
    rule_version: int
    timestamp: datetime
    input_data: dict
    output_data: Any
    success: bool
    execution_time_ms: float
    confidence_score: float
    fallback_used: bool
    llm_would_have_said: str | None  # For comparison studies

class RuleLogger:
    """Comprehensive logging for rule system analytics."""

    def __init__(self, log_dir: str):
        """Initialize with log directory."""

    def log_invocation(self, invocation: RuleInvocation):
        """Log a rule invocation."""

    def log_rule_creation(self, rule: Rule, creation_context: dict):
        """Log when LLM creates a new rule."""

    def log_rule_update(self, old_rule: Rule, new_rule: Rule, update_reason: str):
        """Log when LLM updates an existing rule."""

    def log_routing_decision(self, decision: RoutingDecision, actual_outcome: str):
        """Log routing decisions for confidence calibration analysis."""

    # Analytics
    def get_rule_usage_over_time(self, rule_id: str) -> pd.DataFrame:
        """Get time series of rule usage."""

    def get_rule_performance_metrics(self, rule_id: str) -> dict:
        """Get performance metrics for a rule."""

    def get_confidence_calibration(self) -> dict:
        """Analyze how well confidence predicts success."""
```

---

## 4. Integration with RLM

### 4.1 New REPL Functions

Add these functions to the REPL environment (in `local_repl.py`):

```python
# In LocalREPL.setup()
self.globals["rule_query"] = self._rule_query
self.globals["create_rule"] = self._create_rule
self.globals["update_rule"] = self._update_rule
self.globals["list_rules"] = self._list_rules
self.globals["get_rule_stats"] = self._get_rule_stats

def _rule_query(self, query: str, context: dict = None, force_llm: bool = False) -> str:
    """
    Query that routes to rules or LLM based on confidence.

    Args:
        query: The query to answer
        context: Additional context for matching/execution
        force_llm: If True, skip rules and use LLM directly

    Returns:
        Response from rule or LLM, plus metadata about routing decision
    """

def _create_rule(
    self,
    name: str,
    trigger_pattern: str,
    rule_type: str,
    code: str,
    description: str,
    confidence_threshold: float = 0.8
) -> str:
    """
    Create a new rule for future use.

    Returns:
        Rule ID if successful, error message otherwise
    """

def _update_rule(self, rule_id: str, updates: dict, reason: str) -> str:
    """
    Update an existing rule.

    Args:
        rule_id: ID of rule to update
        updates: Dict of fields to update
        reason: Why this update is being made
    """

def _list_rules(self, filter_type: str = None) -> list[dict]:
    """List all available rules, optionally filtered by type."""

def _get_rule_stats(self, rule_id: str = None) -> dict:
    """Get performance statistics for rules."""
```

### 4.2 Modified System Prompt

Update `RLM_SYSTEM_PROMPT` to include rule system instructions:

```python
RULE_SYSTEM_PROMPT_EXTENSION = """
## Rule-Based System

You have access to a rule-based system that can handle certain queries without LLM calls:

### Querying with Rules
Use `rule_query(query, context)` instead of `llm_query()` when:
- The task might match an existing rule
- The task is deterministic and well-defined
- Speed/cost is important

The system will automatically route to rules when confident, or fallback to LLM.

### Creating Rules
When you notice a pattern that could be handled programmatically, create a rule:
```repl
rule_id = create_rule(
    name="descriptive_name",
    trigger_pattern=r"regex pattern for when to apply",
    rule_type="python",  # or "template", "lookup", "transform"
    code="def execute(input): return output",
    description="What this rule does and when to use it",
    confidence_threshold=0.85
)
print(f"Created rule: {rule_id}")
```

### Updating Rules
If a rule isn't working well, update it:
```repl
update_rule(
    rule_id="existing_rule_id",
    updates={"code": "improved code", "confidence_threshold": 0.9},
    reason="Fixed edge case where..."
)
```

### Best Practices
1. Create rules for recurring, deterministic patterns
2. Start with high confidence thresholds (0.85+)
3. Include fallback_to_llm=True for non-critical rules
4. Add clear descriptions explaining rule behavior
5. Review rule stats periodically and update underperforming rules
"""
```

### 4.3 RLM Class Changes

```python
class RLM:
    def __init__(
        self,
        # ... existing params ...
        enable_rules: bool = False,
        rule_store_path: str = "./rules",
        rule_min_confidence: float = 0.7,
        track_rule_counterfactuals: bool = False,  # Compare rule vs LLM
    ):
        # ... existing init ...

        if enable_rules:
            self.rule_store = RuleStore(rule_store_path)
            self.rule_executor = RuleExecutor()
            self.rule_router = RuleRouter(
                self.rule_store,
                min_confidence=rule_min_confidence
            )
            self.rule_logger = RuleLogger(os.path.join(rule_store_path, "logs"))
        else:
            self.rule_store = None
```

---

## 5. Research Experiments

### 5.1 Experiment 1: Rule Emergence

**Objective**: Measure how effectively LLMs create useful rules.

**Setup**:
- Dataset: RULER benchmark or custom long-context tasks with recurring patterns
- Baseline: Standard RLM without rules
- Treatment: RLM with rule system enabled

**Metrics**:
- Number of rules created per 100 queries
- Rule reuse rate (how often created rules are invoked)
- Rule quality score (manual evaluation of rule correctness)
- Types of rules created (distribution across python/template/lookup/transform)

**Analysis**:
- What patterns does the LLM recognize as rule-worthy?
- Are rules generalizable or overly specific?
- How does rule creation vary by task type?

### 5.2 Experiment 2: Confidence Calibration

**Objective**: Evaluate how well confidence scores predict rule success.

**Setup**:
- Run queries through rule router
- Log: (confidence_score, actual_success) pairs
- Compare different confidence models (heuristic vs learned)

**Metrics**:
- Expected Calibration Error (ECE)
- Brier score
- AUROC for success prediction
- Confusion matrix at different thresholds

**Analysis**:
- At what confidence threshold is rule use optimal?
- What features best predict rule success?
- Should confidence be rule-specific or global?

### 5.3 Experiment 3: Performance Trade-offs

**Objective**: Quantify latency, cost, and accuracy implications.

**Setup**:
- Same task set with rules enabled vs disabled
- Track counterfactuals (what LLM would have said)

**Metrics**:
- Latency: rule execution time vs LLM call time
- Cost: Token usage with/without rules
- Accuracy: Task success rate with/without rules
- Rule-LLM agreement rate

**Analysis**:
- What is the speed/cost improvement from rules?
- What is the accuracy cost (if any)?
- Is there a task complexity threshold for rule effectiveness?

### 5.4 Experiment 4: Interpretability Assessment

**Objective**: Evaluate whether rules provide meaningful interpretability.

**Setup**:
- Extract all created rules
- Human evaluation of rule understandability
- Compare rule explanations to LLM chain-of-thought

**Metrics**:
- Interpretability score (1-5 human rating)
- Rule description quality (coherence, completeness)
- Debugging utility (can errors be identified from rules?)
- Prediction accuracy from rules alone (no execution)

**Analysis**:
- Do rules reveal the LLM's decision-making process?
- Are rule descriptions accurate explanations?
- Can non-experts understand rule behavior?

### 5.5 Experiment 5: Rule Evolution

**Objective**: Track how rules improve over iterative updates.

**Setup**:
- Long-running deployment with rule persistence
- Track rule versions over time
- Measure performance before/after updates

**Metrics**:
- Update frequency per rule
- Performance improvement per update
- Rule convergence (do rules stabilize?)
- Update quality (is each update an improvement?)

**Analysis**:
- Do LLMs effectively improve their own rules?
- What triggers updates (failures, inefficiency, etc.)?
- Is there a limit to rule improvement?

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Foundation)
- [ ] Define Rule dataclass and schema
- [ ] Implement RuleStore with JSON backend
- [ ] Implement RuleExecutor with sandboxed execution
- [ ] Basic RuleRouter with heuristic confidence
- [ ] RuleLogger with basic invocation tracking
- [ ] Unit tests for all components

### Phase 2: RLM Integration
- [ ] Add rule functions to LocalREPL
- [ ] Extend system prompt with rule instructions
- [ ] Add rule configuration to RLM class
- [ ] Wire up rule components in completion flow
- [ ] Integration tests with simple rules

### Phase 3: LLM Rule Creation
- [ ] Design prompts for rule creation
- [ ] Implement create_rule() with LLM code generation
- [ ] Add validation for LLM-generated rules
- [ ] Implement update_rule() with version tracking
- [ ] Safety checks for malicious/buggy rules

### Phase 4: Confidence & Routing
- [ ] Implement pattern matching for triggers
- [ ] Build confidence scoring function
- [ ] Add A/B routing logic
- [ ] Implement counterfactual tracking
- [ ] Tune confidence thresholds

### Phase 5: Analytics & Monitoring
- [ ] Build analytics dashboard
- [ ] Implement rule performance reports
- [ ] Add confidence calibration analysis
- [ ] Create rule evolution visualizations
- [ ] Export formats for research analysis

### Phase 6: Experiments
- [ ] Set up experiment infrastructure
- [ ] Run Experiment 1: Rule Emergence
- [ ] Run Experiment 2: Confidence Calibration
- [ ] Run Experiment 3: Performance Trade-offs
- [ ] Run Experiment 4: Interpretability
- [ ] Run Experiment 5: Rule Evolution
- [ ] Statistical analysis and paper writing

---

## 7. Evaluation Datasets

### 7.1 Recommended Benchmarks

| Benchmark | Why Suitable | Rule Opportunity |
|-----------|--------------|------------------|
| **RULER** | Long-context, recurring patterns | Retrieval rules |
| **HotpotQA** | Multi-hop reasoning | Decomposition rules |
| **DROP** | Discrete reasoning | Calculation rules |
| **WikiTableQuestions** | Structured data | Query rules |
| **NaturalQuestions** | Factual lookup | Lookup rules |
| **GSM8K** | Math word problems | Math operation rules |

### 7.2 Custom Evaluation Sets

Create synthetic datasets with known rule-amenable patterns:
- Date parsing/formatting variations
- Unit conversions
- Entity resolution
- Template filling
- Regex extraction
- Lookup table queries

---

## 8. Expected Outcomes

### 8.1 Primary Contributions

1. **Architecture**: First framework for LLM-designed rule systems within recursive LM paradigm
2. **Methodology**: Protocols for studying rule emergence, calibration, and evolution
3. **Empirical Findings**: Quantitative analysis of rule effectiveness across dimensions
4. **Interpretability Insights**: Novel lens into LLM decision-making through self-created rules

### 8.2 Hypotheses

| Hypothesis | Expected Finding |
|------------|------------------|
| H1: Rule Emergence | LLMs create 15-30 rules per 1000 queries with >60% reuse rate |
| H2: Confidence | Heuristic confidence achieves <0.1 ECE with tuning |
| H3: Performance | Rules provide 10-100x latency reduction at <5% accuracy cost |
| H4: Interpretability | Rules score >3.5/5 on human interpretability ratings |
| H5: Evolution | Rules improve by 10-20% per update cycle |

### 8.3 Potential Risks

| Risk | Mitigation |
|------|------------|
| LLM creates buggy/insecure rules | Sandboxed execution, input/output validation |
| Rules overfit to training queries | Generalization testing, pattern diversity analysis |
| Confidence poorly calibrated | Multiple confidence models, threshold tuning |
| Rules don't provide interpretability | Human evaluation, description quality metrics |

---

## 9. File Structure

```
rlm/
├── rules/
│   ├── __init__.py
│   ├── types.py          # Rule, RuleMatch, RuleResult, RoutingDecision
│   ├── store.py          # RuleStore
│   ├── executor.py       # RuleExecutor
│   ├── router.py         # RuleRouter
│   ├── logger.py         # RuleLogger, RuleInvocation
│   └── prompts.py        # Rule-related prompts for LLM
├── core/
│   └── rlm.py            # Updated with rule support
├── environments/
│   └── local_repl.py     # Updated with rule functions
└── utils/
    └── prompts.py        # Updated system prompt

experiments/
├── rule_emergence/
│   ├── run_experiment.py
│   └── analyze_results.py
├── confidence_calibration/
│   ├── run_experiment.py
│   └── analyze_results.py
├── performance_tradeoffs/
│   ├── run_experiment.py
│   └── analyze_results.py
├── interpretability/
│   ├── run_experiment.py
│   ├── human_eval_interface.py
│   └── analyze_results.py
└── rule_evolution/
    ├── run_experiment.py
    └── analyze_results.py

docs/
└── research/
    └── rule-based-system-plan.md  # This document
```

---

## 10. Success Criteria

The research is successful if:

1. **Technical**: Rule system integrates cleanly with RLM, enables LLM rule creation
2. **Empirical**: At least 3/5 hypotheses are supported by experiments
3. **Practical**: Rules provide measurable latency/cost improvements
4. **Scientific**: Results are novel and publishable at top ML venues
5. **Interpretability**: Rules provide genuine insight into LLM behavior

---

## Appendix A: Related Work

- **Tool-augmented LLMs**: Toolformer, Gorilla, API-Bank
- **Program Synthesis**: AlphaCode, Codex, code-as-policy
- **Self-improvement**: Self-Refine, Reflexion, Constitutional AI
- **Interpretability**: Chain-of-thought, tree-of-thought, faithful reasoning
- **Caching/Memoization**: Semantic caching, prompt compression

---

## Appendix B: Questions for Discussion

1. Should rules be shared across RLM instances or per-instance?
2. How do we handle rule conflicts (multiple rules match)?
3. Should there be a "rule review" step before deployment?
4. How do we prevent rule proliferation (too many low-quality rules)?
5. Should rules be typed/categorized beyond the 4 types proposed?
6. How do we handle rule deprecation?
7. Should users be able to inject/override rules?
