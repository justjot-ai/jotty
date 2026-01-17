---
task_id: TASK-STOCK-SCREENER-20260117
title: "Build Stock Market Screening System"
created_at: 2026-01-17
priority: high
estimated_duration: "60-90 minutes"
workflow_pattern: "hybrid"  # P2P discovery + Sequential delivery
agents_required: 6
---

# Task: Build Stock Market Screening System

## Objective

Design and implement a multi-agent stock market screening system that identifies undervalued growth stocks using fundamental analysis.

**Data Location**: `/var/www/sites/personal/stock_market/common/Data/FUNDAMENTALS/`

## Phase 1: P2P Discovery (Parallel Research)

**Goal**: Explore the problem space from multiple angles

### Discovery Agents (Run in Parallel):

#### 1. **Financial Data Analyst**
- **Focus**: Analyze available data files
- **Tasks**:
  - Examine `BalanceSheet_data.xlsx`, `PnL_data.xlsx`, `Cashflow_data.xlsx`
  - Identify key financial metrics (ROE, P/E, Debt/Equity, Revenue Growth, etc.)
  - Determine data quality and completeness
  - Recommend which metrics to use for screening

#### 2. **Ratio & Valuation Expert**
- **Focus**: Define screening criteria
- **Tasks**:
  - Analyze `Ratio_data.xlsx` and `Quarterly_data.xlsx`
  - Identify undervaluation indicators (P/E < industry avg, P/B < 1, etc.)
  - Identify growth indicators (Revenue CAGR > 15%, EPS growth, etc.)
  - Define score/ranking methodology

#### 3. **Technical Analyst**
- **Focus**: Market trends and momentum
- **Tasks**:
  - Analyze `Technical.csv` data
  - Identify momentum indicators (RSI, MACD, moving averages)
  - Define entry/exit signals
  - Recommend filters to avoid value traps

#### 4. **System Architect**
- **Focus**: Technical implementation approach
- **Tasks**:
  - Design data ingestion pipeline (Excel → processing)
  - Recommend Python libraries (pandas, numpy, yfinance if needed)
  - Define screening algorithm architecture
  - Plan for scalability and performance

**Deliverable from Phase 1**:
- Shared insights on what metrics to use
- Consensus on screening criteria
- Technical approach recommendations

---

## Phase 2: Sequential Delivery (Ordered Build)

**Goal**: Build the system using discoveries from Phase 1

Use **Hybrid Template** pattern with discoveries from Phase 1 agents.

### Delivery Agents (Run Sequentially):

#### 1. **Requirements Engineer** (First)
- **Input**: All discoveries from Phase 1
- **Output**:
  - Complete PRD for stock screener
  - User stories and acceptance criteria
  - Success metrics (precision, recall, returns)
  - API/CLI interface specification

#### 2. **Data Engineer** (Builds on Requirements)
- **Input**: PRD + Phase 1 data analysis
- **Output**:
  - Data loading modules (read Excel files)
  - Data cleaning and normalization pipeline
  - Feature engineering (calculate ratios, growth rates)
  - Data validation and quality checks
  - **Code**: `core/stock_screener/data_loader.py`

#### 3. **Screening Engine Developer** (Builds on Data)
- **Input**: Data pipeline + screening criteria
- **Output**:
  - Screening algorithm implementation
  - Multi-criteria filtering (valuation + growth + technical)
  - Scoring and ranking system
  - **Code**: `core/stock_screener/screening_engine.py`

#### 4. **Backend Developer** (Builds on Engine)
- **Input**: Screening engine + requirements
- **Output**:
  - CLI interface for running screens
  - API endpoints (if web-based)
  - Result formatting (CSV, JSON, HTML reports)
  - **Code**: `core/stock_screener/api.py`

#### 5. **Test Engineer** (Validates Everything)
- **Input**: All components above
- **Output**:
  - Unit tests for data loader
  - Integration tests for screening engine
  - Backtesting framework (historical performance)
  - Test data fixtures
  - **Code**: `tests/stock_screener/`

#### 6. **Documentation Writer** (Final)
- **Input**: Complete system
- **Output**:
  - README with usage examples
  - API documentation
  - Screening methodology explanation
  - Deployment guide
  - **File**: `docs/STOCK_SCREENER.md`

---

## Expected Outputs

### Code Files
```
core/stock_screener/
├── __init__.py
├── data_loader.py         # Load and clean Excel files
├── metrics.py             # Calculate financial ratios
├── screening_engine.py    # Multi-criteria screening
├── api.py                 # CLI/API interface
└── reports.py             # Generate output reports

tests/stock_screener/
├── test_data_loader.py
├── test_metrics.py
├── test_screening_engine.py
└── fixtures/              # Test data
```

### Documentation
- `docs/STOCK_SCREENER.md` - Complete guide
- `docs/SCREENING_CRITERIA.md` - Methodology explanation
- `examples/screening_example.py` - Usage examples

### Deliverables
- Working CLI: `python -m core.stock_screener --criteria "undervalued_growth"`
- Sample output: Top 20 stocks ranked by score
- Backtesting results: Historical performance

---

## Data Available

Located at: `/var/www/sites/personal/stock_market/common/Data/FUNDAMENTALS/`

| File | Size | Description |
|------|------|-------------|
| `BalanceSheet_data.xlsx` | 2.5M | Assets, liabilities, equity |
| `PnL_data.xlsx` | 3.4M | Revenue, expenses, profit |
| `Cashflow_data.xlsx` | 1.5M | Operating, investing, financing cash flows |
| `Ratio_data.xlsx` | 1.7M | Financial ratios (P/E, P/B, ROE, etc.) |
| `Quarterly_data.xlsx` | 3.0M | Quarterly financials |
| `Technical.csv` | 14M | Price, volume, technical indicators |
| `Equity.csv` | 814K | Stock list and metadata |
| `trainingdata.csv` | 6.1M | Historical training data |

---

## Success Criteria

1. **Functional**:
   - System loads all data files successfully
   - Screening produces ranked list of stocks
   - Results are reproducible

2. **Quality**:
   - Code follows best practices
   - Comprehensive test coverage (>80%)
   - Clear documentation

3. **Performance**:
   - Screening completes in < 30 seconds
   - Handles full dataset (all stocks)

4. **Accuracy**:
   - Backtesting shows positive alpha
   - Low false positives (value traps filtered out)

---

## Collaboration Instructions

### Use Hybrid Workflow Template

```python
# Run this task using hybrid template
python templates/hybrid_team_template.py
```

**Phase 1**: 4 discovery agents run in parallel
- Share findings via SharedScratchpad
- Build consensus on approach

**Phase 2**: 6 delivery agents run sequentially
- Each builds on previous
- Use discoveries from Phase 1

### Shared Workspace

All agents should:
- ✅ Use `SharedContext` to store intermediate results
- ✅ Post insights to `SharedScratchpad`
- ✅ Read messages from other agents
- ✅ Session persisted to `workspace/scratchpads/stock_screener.jsonl`

---

## Execution Command

```bash
# Run with Jotty multi-agent system
cd /var/www/sites/personal/stock_market/Jotty

# Execute task
python -m core.orchestration.task_executor \
  --task tasks/TASK-STOCK-SCREENER.md \
  --workflow hybrid \
  --session stock_screener_$(date +%Y%m%d)
```

---

## Notes

- **Real Data**: Use actual Excel/CSV files from FUNDAMENTALS directory
- **Real Code**: Agents should generate actual Python code files
- **Real Tests**: Write executable test files
- **Real Collaboration**: Agents communicate via scratchpad, not just string passing

This is a **meta-task**: Jotty agents building a stock screening system! 🚀
