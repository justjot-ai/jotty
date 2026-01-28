# Paytm Stock Research Analysis

## Test Question
"Research Paytm Stocks on Fundamental, Technicals basis and summarize report as PDF"

## Current Behavior ❌

### What Happened
- **Task Type Detected**: `ANALYSIS`
- **Skills Discovered**: 5/7 expected skills
- **Skills Actually Used**: Only 2 (web-search + summarize)
- **PDF Generated**: ❌ No
- **Stock Research Skill Used**: ❌ No

### Skills Discovered But NOT Used
- ✅ `stock-research-comprehensive` - Discovered but NOT used
- ✅ `stock-research-deep` - Discovered but NOT used  
- ✅ `screener-financials` - Discovered but NOT used
- ✅ `pdf-tools` - Discovered but NOT used
- ⚠️ `research-to-pdf` - Not discovered
- ⚠️ `claude-cli-llm` - Not discovered (but summarize skill was used)

### What Was Actually Executed
1. Web search for "Research Paytm Stocks..."
2. Summarize search results
3. **STOPPED** - No PDF, no comprehensive stock research

## Expected Behavior ✅

### What Should Happen
1. **Detect Stock Research Request**
   - Keywords: "Paytm", "Stocks", "Fundamental", "Technicals"
   - Use `stock-research-comprehensive` skill

2. **Execute Comprehensive Stock Research**
   - 12 parallel web searches (fundamentals, technicals, broker reports)
   - LLM synthesis (claude-cli-llm)
   - Generate markdown report

3. **Generate PDF**
   - Convert markdown to PDF
   - Save to output directory

4. **Skills Should Be Used**
   - `stock-research-comprehensive` (orchestrates everything)
   - `web-search` (12 searches)
   - `claude-cli-llm` (synthesis)
   - `document-converter` (PDF generation)
   - Potentially `lida-to-justjot` (if visualizations needed)

## Root Cause Analysis

### Problem 1: Task Type Detection
**Location**: `core/agents/auto_agent.py` - `_infer_task_type()`

**Issue**: 
- Request detected as `ANALYSIS` (because of "analyze" keyword)
- `_plan_analysis()` just calls `_plan_research()`
- No special handling for stock research

**Fix Needed**:
```python
# Add stock research detection
if any(w in task_lower for w in ['stock', 'ticker', 'fundamental', 'technical', 'equity', 'share']):
    # Check if it's a stock research request
    if any(w in task_lower for w in ['research', 'analyze', 'analysis']):
        return TaskType.STOCK_RESEARCH  # New task type
```

### Problem 2: Planning Logic
**Location**: `core/agents/auto_agent.py` - `_plan_analysis()`

**Issue**:
- `_plan_analysis()` doesn't check for stock research
- Falls back to generic research plan
- Doesn't use `stock-research-comprehensive` skill

**Fix Needed**:
```python
def _plan_analysis(self, task: str, skills: List[str]) -> List[ExecutionStep]:
    """Plan for analysis tasks."""
    # Check if it's stock research
    if self._is_stock_research(task):
        return self._plan_stock_research(task, skills)
    # Otherwise, use generic research
    return self._plan_research(task, skills)

def _is_stock_research(self, task: str) -> bool:
    """Check if task is stock research."""
    task_lower = task.lower()
    stock_keywords = ['stock', 'ticker', 'equity', 'share', 'fundamental', 'technical']
    return any(kw in task_lower for kw in stock_keywords)

def _plan_stock_research(self, task: str, skills: List[str]) -> List[ExecutionStep]:
    """Plan for stock research tasks."""
    steps = []
    
    # Use stock-research-comprehensive if available
    if 'stock-research-comprehensive' in skills:
        steps.append(ExecutionStep(
            skill_name='stock-research-comprehensive',
            tool_name='comprehensive_stock_research_tool',
            params={
                'ticker': self._extract_ticker(task),
                'company_name': self._extract_company_name(task),
                'send_telegram': False
            },
            description='Comprehensive stock research (fundamentals + technicals + broker reports)',
            output_key='stock_research_result'
        ))
    # ... fallback to other stock research skills
```

### Problem 3: Skill Discovery
**Location**: `core/agents/auto_agent.py` - `_discover_skills()`

**Issue**:
- Skill discovery finds `stock-research-comprehensive`
- But planning doesn't prioritize it for stock research tasks
- Generic planning logic doesn't recognize stock research pattern

**Fix Needed**:
- Enhance skill discovery to prioritize domain-specific skills
- Add stock research pattern matching

## Solution: Enhanced Stock Research Detection

### Option 1: Add Stock Research Task Type (Recommended)

```python
class TaskType(Enum):
    RESEARCH = "research"
    COMPARISON = "comparison"
    CREATION = "creation"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    STOCK_RESEARCH = "stock_research"  # NEW
    UNKNOWN = "unknown"
```

### Option 2: Enhance Analysis Planning

```python
def _plan_analysis(self, task: str, skills: List[str]) -> List[ExecutionStep]:
    """Plan for analysis tasks."""
    task_lower = task.lower()
    
    # Check for stock research
    if any(kw in task_lower for kw in ['stock', 'ticker', 'fundamental', 'technical']):
        return self._plan_stock_research(task, skills)
    
    # Check for data analysis (with LIDA)
    if any(kw in task_lower for kw in ['visualize', 'chart', 'graph', 'dashboard']):
        return self._plan_data_analysis(task, skills)
    
    # Default: research flow
    return self._plan_research(task, skills)
```

## Expected Workflow After Fix

### Request: "Research Paytm Stocks on Fundamental, Technicals basis and summarize report as PDF"

1. **Intent Parser**
   - Detects: stock research request
   - Extracts: ticker="PAYTM", company="Paytm"
   - Task Type: `STOCK_RESEARCH`

2. **Planner**
   - Discovers: `stock-research-comprehensive` skill
   - Plans: Use comprehensive stock research tool
   - Adds: PDF generation step

3. **Executor**
   - Executes: `stock-research-comprehensive` skill
     - 12 parallel web searches
     - LLM synthesis
     - Markdown generation
     - PDF conversion
   - Result: Comprehensive PDF report

4. **Output**
   - Markdown file: `output/paytm_comprehensive_research.md`
   - PDF file: `output/paytm_comprehensive_research.pdf`
   - Contains: Fundamentals + Technicals + Broker Reports

## Skills That Should Be Used

1. **stock-research-comprehensive** ✅ (Main orchestrator)
   - Uses: web-search (12 searches)
   - Uses: claude-cli-llm (synthesis)
   - Uses: document-converter (PDF)
   - Uses: file-operations (file management)

2. **lida-to-justjot** ⚠️ (Optional - if visualizations requested)
   - Could add charts for technical analysis

3. **research-to-pdf** ⚠️ (Alternative - if stock-research-comprehensive not available)

## Conclusion

**Current State**: ❌ Agent discovers skills but doesn't use them properly for stock research

**Root Cause**: 
- No stock research detection in task type inference
- Generic analysis planning doesn't recognize stock research pattern
- Planning logic doesn't prioritize domain-specific skills

**Solution**: 
- Add stock research task type detection
- Enhance planning logic to use `stock-research-comprehensive` for stock research requests
- Chain PDF generation automatically

**Expected After Fix**: ✅ Agent will use `stock-research-comprehensive` → generates comprehensive PDF with fundamentals + technicals + broker reports
