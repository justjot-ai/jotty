# GAIA Benchmark - Fundamental Fixes Plan

**Goal:** Perfect GAIA scores through fundamental architectural improvements
**Approach:** No hacks - fix root causes

---

## 🔍 Root Cause Analysis

### Current Hacks (TO REMOVE):
1. `skip_swarm_selection=True` - Bypasses broken swarm selector
2. `skip_complexity_gate=True` - Bypasses broken complexity detection
3. Manual skill hints - Workaround for poor auto-detection

### Fundamental Problems:

#### 1. **Swarm Selection is Broken**
**Current:** Keyword matching `"report" → ReviewSwarm`
**Problem:** GAIA prompts contain misleading keywords
**Root Cause:** No semantic understanding of task intent

#### 2. **Wrong Architecture for Q&A Tasks**
**Current:** Routes through domain swarms (coding, research, etc.)
**Problem:** Swarms are for complex workflows, not direct Q&A
**Root Cause:** No dedicated path for fact-retrieval tasks

#### 3. **Tool Access is Indirect**
**Current:** Tools only available through specific swarms
**Problem:** Can't access calculator from ReviewSwarm
**Root Cause:** Tool registry not directly accessible

#### 4. **No Structured Output**
**Current:** Verbose swarm outputs with reasoning
**Problem:** GAIA needs exact answers only
**Root Cause:** No answer extraction layer

#### 5. **No Multi-Step Planning**
**Current:** Single execution with blind retries
**Problem:** Complex questions need decomposition
**Root Cause:** No task planner for multi-hop reasoning

---

## 🏗️ Fundamental Fix #1: Intent-Based Routing

### Replace Keyword Matching with Intent Classification

**Current (Broken):**
```python
def _select_swarm(self, goal: str, swarm_name: Optional[str] = None):
    if "report" in goal.lower():
        return ReviewSwarm()  # WRONG!
```

**Fundamental Fix:**
```python
class TaskIntentClassifier:
    """Classify task intent using LLM-based understanding."""

    def classify(self, task: str, attachments: List[str]) -> TaskIntent:
        """
        Returns one of:
        - FACT_RETRIEVAL: Direct Q&A (GAIA tasks)
        - CODE_GENERATION: Write/debug code
        - CONTENT_CREATION: Write articles/reports
        - DATA_ANALYSIS: Analyze datasets
        - RESEARCH: Deep multi-source research
        """
        # Use LLM to classify based on task structure, not keywords
        prompt = f"""Classify this task intent:

Task: {task}
Attachments: {attachments}

Choose ONE:
1. FACT_RETRIEVAL - Answer a specific question (who/what/when/where/calculate)
2. CODE_GENERATION - Write, debug, or review code
3. CONTENT_CREATION - Write articles, reports, documents
4. DATA_ANALYSIS - Analyze data, create visualizations
5. RESEARCH - Multi-source research with synthesis

Answer with just the category name."""

        response = self.lm(prompt)
        return TaskIntent(response.strip())

    def requires_tools(self, intent: TaskIntent, task: str) -> List[str]:
        """Determine which tools are needed."""
        if intent == TaskIntent.FACT_RETRIEVAL:
            # Analyze what tools are needed
            needs = []
            if "calculate" in task or "math" in task or any(op in task for op in ['+','-','*','/']):
                needs.append('calculator')
            if "search" in task or "find" in task or "what is" in task:
                needs.append('web-search')
            if any(ext in task for ext in ['.mp3', '.wav', 'audio']):
                needs.append('whisper')
            if any(ext in task for ext in ['.pdf', '.docx', 'document']):
                needs.append('document-reader')
            return needs
        # ... other intents
```

**Implementation:**
- Create `Jotty/core/execution/intent_classifier.py`
- Replace `_select_swarm()` with `_classify_and_route()`
- Use LLM for semantic understanding, not keywords

---

## 🏗️ Fundamental Fix #2: GAIA-Optimized Executor

### Create Dedicated Path for Fact-Retrieval Tasks

**New Architecture:**
```python
class FactRetrievalExecutor:
    """Optimized executor for direct Q&A tasks like GAIA."""

    def __init__(self, registry: UnifiedRegistry):
        self.registry = registry
        self.planner = MultiStepPlanner()
        self.answer_extractor = AnswerExtractor()

    async def execute(self, question: str, attachments: List[str]) -> str:
        """Execute fact-retrieval task with perfect accuracy."""

        # Step 1: Analyze question structure
        analysis = self._analyze_question(question)

        # Step 2: Decompose into steps if needed
        if analysis.is_multi_hop:
            steps = self.planner.decompose(question)
        else:
            steps = [question]

        # Step 3: Execute each step with appropriate tools
        context = {}
        for step in steps:
            tools_needed = self._identify_tools(step, attachments)
            result = await self._execute_step(step, tools_needed, context)
            context[step] = result

        # Step 4: Extract exact answer
        final_answer = self.answer_extractor.extract(
            question=question,
            context=context,
            expected_format=analysis.answer_format
        )

        return final_answer

    def _analyze_question(self, question: str) -> QuestionAnalysis:
        """Understand question structure and requirements."""
        return QuestionAnalysis(
            is_multi_hop=self._is_multi_hop(question),
            answer_format=self._detect_format(question),
            domain=self._detect_domain(question),
            tools_needed=self._predict_tools(question)
        )

    def _is_multi_hop(self, question: str) -> bool:
        """Detect if question requires multiple reasoning steps."""
        # "What is the capital of the country where X lives?"
        # → Step 1: Where does X live?
        # → Step 2: What's the capital of that country?

        indicators = [
            "where.*live.*capital",  # Multi-hop location
            "who.*created.*when",     # Multi-hop entity
            "how many.*in.*that",     # Multi-hop counting
        ]
        return any(re.search(pattern, question.lower()) for pattern in indicators)

    def _detect_format(self, question: str) -> AnswerFormat:
        """Detect expected answer format."""
        if re.search(r"how many|number of", question.lower()):
            return AnswerFormat.NUMBER
        if re.search(r"when|what year|what date", question.lower()):
            return AnswerFormat.DATE
        if re.search(r"who|person|people", question.lower()):
            return AnswerFormat.PERSON
        if re.search(r"where|location|place", question.lower()):
            return AnswerFormat.LOCATION
        if re.search(r"calculate|compute|sum|multiply", question.lower()):
            return AnswerFormat.NUMBER
        return AnswerFormat.TEXT

    async def _execute_step(self, step: str, tools: List[str], context: Dict) -> Any:
        """Execute single step with tools."""
        # Get tools from registry
        available_tools = [self.registry.get_skill(t) for t in tools]

        # Create single-purpose agent with just these tools
        agent = SimpleQAAgent(tools=available_tools, context=context)

        # Execute with strict output requirements
        result = await agent.execute(step, max_tokens=500, require_exact=True)

        return result
```

**Key Principles:**
1. **Direct tool access** - No swarm indirection
2. **Multi-step decomposition** - Handle complex reasoning
3. **Answer extraction** - Get exact output
4. **Format awareness** - Return in expected format

---

## 🏗️ Fundamental Fix #3: Multi-Step Planner

### Proper Task Decomposition

**Current:** Single execution, fails on multi-hop
**Fix:** Decompose complex questions into atomic steps

```python
class MultiStepPlanner:
    """Decompose multi-hop questions into atomic steps."""

    def decompose(self, question: str) -> List[Step]:
        """Break down question into answerable sub-questions."""

        # Example: "What is the capital of the country where Rockhopper penguins live?"
        # → Step 1: "Where do Rockhopper penguins live?"
        # → Step 2: "What is the capital of {answer from step 1}?"

        prompt = f"""Decompose this question into step-by-step sub-questions.
Each step should be answerable independently.

Question: {question}

Provide steps in this format:
Step 1: [atomic question]
Step 2: [atomic question that may reference Step 1]
...

Steps:"""

        response = self.lm(prompt, temperature=0.0)  # Deterministic
        steps = self._parse_steps(response)

        return [Step(text=s, depends_on=self._extract_deps(s)) for s in steps]

    def _extract_deps(self, step_text: str) -> List[int]:
        """Extract which previous steps this depends on."""
        # "What is the capital of {Step 1}?"  →  depends_on=[1]
        deps = []
        for match in re.finditer(r"\{Step (\d+)\}", step_text):
            deps.append(int(match.group(1)))
        return deps
```

---

## 🏗️ Fundamental Fix #4: Answer Extractor

### Structured Output Parsing

**Current:** Verbose reasoning mixed with answer
**Fix:** Extract exact answer in expected format

```python
class AnswerExtractor:
    """Extract exact answers from verbose outputs."""

    def extract(
        self,
        question: str,
        context: Dict[str, Any],
        expected_format: AnswerFormat
    ) -> str:
        """Extract exact answer matching expected format."""

        # Combine all context
        full_context = "\n\n".join(f"Step: {k}\nResult: {v}" for k, v in context.items())

        prompt = f"""Extract the exact answer to this question from the context.

Question: {question}
Expected format: {expected_format.value}

Context:
{full_context}

Provide ONLY the answer, nothing else. No explanation, no reasoning.
Format: {self._format_example(expected_format)}

Answer:"""

        response = self.lm(prompt, temperature=0.0, max_tokens=50)
        answer = response.strip()

        # Validate format
        if not self._validate_format(answer, expected_format):
            # Try to fix common issues
            answer = self._fix_format(answer, expected_format)

        return answer

    def _validate_format(self, answer: str, format: AnswerFormat) -> bool:
        """Verify answer matches expected format."""
        if format == AnswerFormat.NUMBER:
            try:
                float(answer.replace(',', ''))
                return True
            except:
                return False
        elif format == AnswerFormat.DATE:
            # Check for date-like patterns
            date_patterns = [
                r'\d{4}',  # Year
                r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                r'[A-Z][a-z]+ \d{1,2}, \d{4}',  # Month DD, YYYY
            ]
            return any(re.match(p, answer) for p in date_patterns)
        # ... other formats
        return True

    def _fix_format(self, answer: str, format: AnswerFormat) -> str:
        """Fix common formatting issues."""
        if format == AnswerFormat.NUMBER:
            # Extract first number found
            match = re.search(r'-?\d+\.?\d*', answer)
            if match:
                return match.group()
        elif format == AnswerFormat.LOCATION:
            # Remove "in", "at", etc.
            answer = re.sub(r'^(in|at|near)\s+', '', answer, flags=re.IGNORECASE)
        # ... other fixes
        return answer
```

---

## 🏗️ Fundamental Fix #5: Tool Execution Layer

### Direct, Reliable Tool Access

**Current:** Tools buried in swarms
**Fix:** Direct registry access with fallbacks

```python
class ToolExecutor:
    """Reliable tool execution with fallbacks."""

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        max_retries: int = 3
    ) -> Any:
        """Execute tool with automatic fallbacks."""

        tool = self.registry.get_skill(tool_name)

        for attempt in range(max_retries):
            try:
                result = await tool.execute(**params)

                if self._is_valid_result(result):
                    return result
                else:
                    # Try with modified params
                    params = self._adjust_params(params, attempt)

            except Exception as e:
                logger.warning(f"Tool {tool_name} attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Try fallback strategy
                    params = self._fallback_strategy(tool_name, params, e)
                else:
                    raise

        raise ToolExecutionError(f"Tool {tool_name} failed after {max_retries} attempts")

    def _fallback_strategy(self, tool_name: str, params: Dict, error: Exception) -> Dict:
        """Adjust strategy based on failure."""
        if tool_name == 'web-search':
            # Try different search queries
            if 'query' in params:
                original = params['query']
                # Add quotes for exact match
                params['query'] = f'"{original}"'
                # Or try synonyms
                # Or try breaking into smaller searches
        elif tool_name == 'calculator':
            # Try different expression formats
            if 'expression' in params:
                expr = params['expression']
                # Remove units
                expr = re.sub(r'[a-zA-Z]+', '', expr)
                params['expression'] = expr

        return params
```

---

## 🏗️ Implementation Plan

### Phase 1: Intent Classification (Week 1)
- [ ] Create `IntentClassifier` class
- [ ] Train/configure LLM-based classifier
- [ ] Test on 100 sample tasks
- [ ] Replace keyword matching in executor
- [ ] **Target:** 95%+ classification accuracy

### Phase 2: Fact-Retrieval Executor (Week 2)
- [ ] Create `FactRetrievalExecutor` class
- [ ] Implement question analysis
- [ ] Integrate with tool registry
- [ ] Test on simple GAIA tasks
- [ ] **Target:** 80%+ on single-hop questions

### Phase 3: Multi-Step Planner (Week 3)
- [ ] Create `MultiStepPlanner` class
- [ ] Implement step decomposition
- [ ] Add dependency resolution
- [ ] Test on multi-hop GAIA tasks
- [ ] **Target:** 70%+ on multi-hop questions

### Phase 4: Answer Extraction (Week 4)
- [ ] Create `AnswerExtractor` class
- [ ] Implement format detection
- [ ] Add format validation
- [ ] Test on all GAIA tasks
- [ ] **Target:** 90%+ format correctness

### Phase 5: Tool Reliability (Week 5)
- [ ] Implement fallback strategies
- [ ] Add retry logic with backoff
- [ ] Test tool execution paths
- [ ] Monitor tool success rates
- [ ] **Target:** 95%+ tool success rate

### Phase 6: Integration & Testing (Week 6)
- [ ] Integrate all components
- [ ] Run full GAIA benchmark
- [ ] Identify remaining failures
- [ ] Fix edge cases
- [ ] **Target:** 90%+ GAIA pass rate

### Phase 7: Optimization (Week 7)
- [ ] Optimize for speed
- [ ] Reduce token usage
- [ ] Improve caching
- [ ] Final benchmark run
- [ ] **Target:** 95%+ GAIA pass rate

### Phase 8: Perfection (Week 8)
- [ ] Analyze all failures
- [ ] Add special handlers for edge cases
- [ ] Final testing
- [ ] **Target:** 98-100% GAIA pass rate

---

## 🎯 Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Intent Classification | Accuracy | 95%+ |
| Single-hop Q&A | Pass rate | 80%+ |
| Multi-hop Q&A | Pass rate | 70%+ |
| Answer Format | Correctness | 90%+ |
| Tool Execution | Success rate | 95%+ |
| **Overall GAIA** | **Pass rate** | **95-100%** |

---

## 🚫 Non-Negotiables

1. **No Hacks** - Remove all `skip_*` flags
2. **Semantic Understanding** - Use LLM for intent, not keywords
3. **Direct Tool Access** - No swarm indirection
4. **Multi-Step Planning** - Proper decomposition
5. **Exact Answers** - Structured extraction
6. **Reliability** - Fallbacks and retries
7. **Testability** - Every component unit tested
8. **Observability** - Log every decision

---

## 📝 Next Steps

1. **Review this plan** - Validate approach
2. **Create Phase 1 PR** - Intent classifier
3. **Iterate rapidly** - Weekly releases
4. **Measure constantly** - Run GAIA after each phase
5. **Adjust as needed** - Data-driven improvements

**Timeline:** 8 weeks to 95%+ GAIA scores
**Effort:** ~160 hours (1 engineer full-time)
**Outcome:** Fundamental, production-ready solution

---

**This is the RIGHT way to fix GAIA. No shortcuts.** 🎯
