"""
Streamlit App-Builder Provider for Jotty V2
============================================

Fully open-source app building with Streamlit.
No cloud credentials required - runs 100% locally.

Capabilities:
- Create single-file Streamlit apps
- Generate dashboards, chat interfaces, data apps
- Auto-install streamlit if needed
- Start local dev server

When user says "Build me a chat app" or "Create a dashboard",
this provider generates a complete Streamlit app.
"""

import os
import time
import logging
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base import SkillProvider, SkillCategory, ProviderCapability, ProviderResult

logger = logging.getLogger(__name__)

# Check if streamlit is available
try:
    import streamlit
    STREAMLIT_AVAILABLE = True
    STREAMLIT_VERSION = streamlit.__version__
except ImportError:
    STREAMLIT_AVAILABLE = False
    STREAMLIT_VERSION = None


# =============================================================================
# Streamlit App Templates
# =============================================================================

APP_TEMPLATE = '''"""
{title}

Auto-generated Streamlit app by Jotty V2.
Run with: streamlit run {filename}
"""
import streamlit as st
{imports}

# Page config
st.set_page_config(
    page_title="{title}",
    page_icon="{icon}",
    layout="{layout}"
)

st.title("{title}")
{description_block}

{body}
'''

CHAT_TEMPLATE = '''
# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        # TODO: Integrate with your LLM (OpenAI, Anthropic, etc.)
        response = f"You said: {prompt}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
'''

DASHBOARD_TEMPLATE = '''
# Sidebar controls
with st.sidebar:
    st.header("Settings")
    {sidebar_controls}

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Metrics")
    {metrics_block}

with col2:
    st.subheader("Chart")
    {chart_block}

# Data table
st.subheader("Data")
{table_block}
'''

FORM_TEMPLATE = '''
# Form
with st.form("main_form"):
    st.subheader("Enter Details")
    {form_fields}

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.success("Form submitted!")
        # TODO: Process form data
        st.json({form_data})
'''

# =============================================================================
# Feature Snippets (for editing existing apps)
# =============================================================================

FEATURE_SNIPPETS = {
    'pdf_download': {
        'import': 'import requests',
        'code': '''
                            # Download PDF button
                            arxiv_id = paper.entry_id.split('/')[-1]
                            if st.button(f" Download PDF", key=f"dl_{i}"):
                                with st.spinner("Downloading..."):
                                    try:
                                        response = requests.get(paper.pdf_url)
                                        if response.status_code == 200:
                                            st.download_button(
                                                label=" Save PDF",
                                                data=response.content,
                                                file_name=f"{arxiv_id}.pdf",
                                                mime="application/pdf",
                                                key=f"save_{i}"
                                            )
                                        else:
                                            st.error("Failed to fetch PDF")
                                    except Exception as e:
                                        st.error(f"Download error: {e}")''',
        'insert_after': 'st.markdown(f"[arXiv]({paper.entry_id})")',
        'keywords': ['download', 'pdf', 'save'],
    },
    'export_csv': {
        'import': '',
        'code': '''
# Export to CSV
if st.button(" Export to CSV"):
    csv = df.to_csv(index=False)
    st.download_button(
        label=" Download CSV",
        data=csv,
        file_name="export.csv",
        mime="text/csv"
    )''',
        'insert_after': 'st.dataframe',
        'keywords': ['export', 'csv', 'download data'],
    },
}

ARXIV_SEARCH_TEMPLATE = '''
# Sidebar filters
with st.sidebar:
    st.header("Search Filters")
    search_query = st.text_input("Search Query", value="machine learning")
    max_results = st.slider("Max Results", 5, 50, 20)
    sort_by = st.selectbox("Sort By", ["relevance", "lastUpdatedDate", "submittedDate"])

    categories = st.multiselect(
        "Categories",
        ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.NE", "cs.IR"],
        default=["cs.AI", "cs.LG"]
    )

# Search button
if st.button(" Search arXiv", type="primary") or search_query:
    with st.spinner("Searching arXiv..."):
        try:
            # Build query with categories
            if categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                full_query = f"({search_query}) AND ({cat_query})"
            else:
                full_query = search_query

            # Search arXiv
            sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }[sort_by]

            client = arxiv.Client()
            search = arxiv.Search(
                query=full_query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results = list(client.results(search))

            if results:
                st.success(f"Found {len(results)} papers")

                # Display results
                for i, paper in enumerate(results):
                    with st.expander(f" {paper.title}", expanded=(i < 3)):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Authors:** {', '.join([a.name for a in paper.authors[:5]])}")
                            if len(paper.authors) > 5:
                                st.markdown(f"*...and {len(paper.authors) - 5} more*")

                            st.markdown(f"**Published:** {paper.published.strftime('%Y-%m-%d')}")
                            st.markdown(f"**Categories:** {', '.join(paper.categories)}")

                        with col2:
                            st.markdown(f"[PDF]({paper.pdf_url})")
                            st.markdown(f"[arXiv]({paper.entry_id})")

                        st.markdown("---")
                        st.markdown("**Abstract:**")
                        st.markdown(paper.summary)

                        # Citation
                        if st.button(f" Copy BibTeX", key=f"cite_{i}"):
                            bibtex = f"""@article{{{paper.entry_id.split('/')[-1]},
    title={{{paper.title}}},
    author={{{' and '.join([a.name for a in paper.authors])}}},
    year={{{paper.published.year}}},
    journal={{arXiv preprint arXiv:{paper.entry_id.split('/')[-1]}}}
}}"""
                            st.code(bibtex, language="bibtex")

                # Summary table
                st.subheader(" Results Summary")
                df = pd.DataFrame([{
                    "Title": p.title[:60] + "..." if len(p.title) > 60 else p.title,
                    "Authors": ", ".join([a.name for a in p.authors[:2]]),
                    "Year": p.published.year,
                    "Categories": ", ".join(p.categories[:2]),
                } for p in results])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No papers found. Try different search terms.")

        except Exception as e:
            st.error(f"Search error: {e}")
            st.info("Make sure to install: pip install arxiv")
'''

STOCK_ANALYSIS_TEMPLATE = '''
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Sidebar
with st.sidebar:
    ticker = st.text_input("Stock Ticker", value="AAPL")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

if ticker:
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if not df.empty:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[0]
            change = ((current_price - prev_price) / prev_price) * 100

            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Change", f"{change:.2f}%", delta=f"{change:.2f}%")
            col3.metric("High", f"${df['High'].max():.2f}")
            col4.metric("Low", f"${df['Low'].min():.2f}")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            ))
            fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.subheader("Historical Data")
            st.dataframe(df.tail(20), use_container_width=True)
        else:
            st.warning(f"No data found for {ticker}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
'''


@dataclass
class StreamlitApp:
    """Represents a Streamlit app."""
    name: str
    path: Path
    main_file: str = "app.py"
    server_process: Optional[subprocess.Popen] = None
    server_port: int = 8501

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.server_process is not None and self.server_process.poll() is None


class StreamlitProvider(SkillProvider):
    """
    Provider for building apps using Streamlit.

    Streamlit is a fully open-source Python framework for building
    data apps and AI interfaces. No cloud credentials required.

    Features:
    - Single-file apps (easy to understand and modify)
    - Built-in chat interface support
    - Automatic hot-reload during development
    - Runs 100% locally

    Usage:
        provider = StreamlitProvider()
        await provider.initialize()

        # Create a new app
        result = await provider.execute("Build a stock analysis dashboard")

        # App is created at ./streamlit_apps/stock_dashboard/app.py
        # Run with: streamlit run app.py
    """

    name = "streamlit"
    version = "1.0.0"
    description = "Build apps with Streamlit - fully open source, no cloud required"

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize StreamlitProvider.

        Args:
            config: Provider configuration
                - workspace_dir: Directory for apps (default: ./streamlit_apps)
                - default_port: Default dev server port (default: 8501)
        """
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.APP_BUILDING,
                actions=["create_app", "generate_chat", "generate_dashboard", "serve"],
                max_concurrent=3,
                requires_network=False,  # Runs locally!
                estimated_latency_ms=2000,
            ),
        ]

        # Configuration
        self.workspace_dir = Path(config.get('workspace_dir', './streamlit_apps')) if config else Path('./streamlit_apps')
        self.default_port = config.get('default_port', 8501) if config else 8501

        # Active apps
        self._apps: Dict[str, StreamlitApp] = {}
        self._current_app: Optional[StreamlitApp] = None

    async def initialize(self) -> bool:
        """
        Initialize Streamlit provider.

        Checks for streamlit installation and auto-installs if needed.

        Returns:
            True if initialization successful
        """
        try:
            if not STREAMLIT_AVAILABLE:
                logger.info("Streamlit not found, installing...")
                install_result = await self._install_streamlit()
                if not install_result:
                    logger.warning("Could not install streamlit")
                    self.is_available = False
                    return False

            # Create workspace directory
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

            self.is_initialized = True
            self.is_available = True

            version = STREAMLIT_VERSION or "installed"
            logger.info(f"Streamlit provider initialized (v{version})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Streamlit provider: {e}")
            self.is_initialized = True
            self.is_available = False
            return False

    async def _install_streamlit(self) -> bool:
        """Install streamlit package."""
        try:
            process = await asyncio.create_subprocess_exec(
                'pip', 'install', 'streamlit',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

            if process.returncode == 0:
                logger.info("Successfully installed streamlit")
                return True
            else:
                logger.warning(f"Failed to install streamlit: {stderr.decode()}")
                return False

        except asyncio.TimeoutError:
            logger.warning("Streamlit installation timed out")
            return False
        except Exception as e:
            logger.warning(f"Could not install streamlit: {e}")
            return False

    def get_categories(self) -> List[SkillCategory]:
        """Get supported skill categories."""
        return [SkillCategory.APP_BUILDING]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """
        Execute a Streamlit app creation task.

        Args:
            task: Natural language task description
            context: Additional context

        Returns:
            ProviderResult with execution output
        """
        start_time = time.time()
        context = context or {}
        task_lower = task.lower()

        try:
            # Check for edit/modify requests first
            if any(kw in task_lower for kw in ['add', 'edit', 'modify', 'update', 'include']):
                result = await self._edit_app(task, context)
            # Route to appropriate handler for new apps
            elif any(kw in task_lower for kw in ['arxiv', 'paper', 'research', 'academic', 'publication', 'citation']):
                result = await self._create_arxiv_app(task, context)
            elif any(kw in task_lower for kw in ['chat', 'conversation', 'assistant', 'chatbot']):
                result = await self._create_chat_app(task, context)
            elif any(kw in task_lower for kw in ['stock', 'finance', 'trading', 'market']):
                result = await self._create_stock_app(task, context)
            elif any(kw in task_lower for kw in ['dashboard', 'analytics', 'metrics']):
                result = await self._create_dashboard_app(task, context)
            elif any(kw in task_lower for kw in ['form', 'input', 'survey']):
                result = await self._create_form_app(task, context)
            elif any(kw in task_lower for kw in ['serve', 'run', 'start']):
                result = await self._serve(context)
            elif any(kw in task_lower for kw in ['stop', 'shutdown']):
                result = await self._stop_server(context)
            else:
                # Default: create based on detected type
                result = await self._create_app_from_task(task, context)

            result.execution_time = time.time() - start_time
            result.provider_name = self.name
            self.record_execution(result)
            return result

        except Exception as e:
            logger.error(f"Streamlit provider error: {e}")
            result = ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
                retryable=True,
            )
            self.record_execution(result)
            return result

    def _extract_app_name(self, task: str) -> str:
        """Extract app name from task description."""
        import re

        # Look for quoted names
        quoted = re.search(r'["\']([^"\']+)["\']', task)
        if quoted:
            return quoted.group(1).lower().replace(' ', '_')

        # Look for "called X" or "named X"
        named = re.search(r'(?:called|named)\s+(\w+)', task, re.IGNORECASE)
        if named:
            return named.group(1).lower()

        task_lower = task.lower()

        # Domain-specific name mapping
        domain_keywords = {
            'arxiv': 'arxiv_searcher',
            'paper': 'paper_searcher',
            'research': 'research_explorer',
            'stock': 'stock_dashboard',
            'finance': 'finance_app',
            'trading': 'trading_app',
        }
        for keyword, name in domain_keywords.items():
            if keyword in task_lower:
                return name

        # Extract key nouns
        words = task_lower.split()
        keywords = ['app', 'dashboard', 'chat', 'analyzer', 'tool', 'interface', 'searcher']
        for i, word in enumerate(words):
            if word in keywords and i > 0:
                return f"{words[i-1]}_{word}"

        return "streamlit_app"

    async def _create_chat_app(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """Create a chat interface app."""
        app_name = context.get('app_name') or self._extract_app_name(task) or "chat_app"

        imports = []
        body = CHAT_TEMPLATE

        return await self._create_app(
            app_name=app_name,
            title="AI Chat Assistant",
            icon="",
            imports=imports,
            body=body,
            description="Chat with the AI assistant below.",
        )

    async def _create_stock_app(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """Create a stock analysis app."""
        app_name = context.get('app_name') or self._extract_app_name(task) or "stock_dashboard"

        imports = [
            "import yfinance as yf",
            "import pandas as pd",
            "import plotly.graph_objects as go",
        ]
        body = STOCK_ANALYSIS_TEMPLATE

        return await self._create_app(
            app_name=app_name,
            title="Stock Analysis Dashboard",
            icon="",
            layout="wide",
            imports=imports,
            body=body,
            description="Analyze stock data with interactive charts.",
            extra_deps=["yfinance", "plotly"],
        )

    async def _create_arxiv_app(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """Create an arXiv paper search app."""
        app_name = context.get('app_name') or self._extract_app_name(task) or "arxiv_searcher"

        imports = [
            "import arxiv",
            "import pandas as pd",
            "from datetime import datetime",
        ]
        body = ARXIV_SEARCH_TEMPLATE

        return await self._create_app(
            app_name=app_name,
            title="arXiv Paper Search",
            icon="",
            layout="wide",
            imports=imports,
            body=body,
            description="Search and explore academic papers from arXiv.",
            extra_deps=["arxiv", "pandas"],
        )

    async def _create_dashboard_app(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """Create a dashboard app."""
        app_name = context.get('app_name') or self._extract_app_name(task) or "dashboard"

        imports = [
            "import pandas as pd",
            "import numpy as np",
        ]

        sidebar_controls = 'date_range = st.date_input("Date Range", [])'
        metrics_block = '''st.metric("Total Users", "1,234", "+12%")
    st.metric("Revenue", "$5,678", "+8%")'''
        chart_block = '''chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    st.line_chart(chart_data)'''
        table_block = '''df = pd.DataFrame({"Column 1": [1, 2, 3], "Column 2": ["A", "B", "C"]})
st.dataframe(df, use_container_width=True)'''

        body = DASHBOARD_TEMPLATE.format(
            sidebar_controls=sidebar_controls,
            metrics_block=metrics_block,
            chart_block=chart_block,
            table_block=table_block,
        )

        return await self._create_app(
            app_name=app_name,
            title="Analytics Dashboard",
            icon="",
            layout="wide",
            imports=imports,
            body=body,
            description="View your metrics and analytics.",
        )

    async def _create_form_app(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """Create a form app."""
        app_name = context.get('app_name') or self._extract_app_name(task) or "form_app"

        imports = []

        form_fields = '''name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")'''
        form_data = '{"name": name, "email": email, "message": message}'

        body = FORM_TEMPLATE.format(
            form_fields=form_fields,
            form_data=form_data,
        )

        return await self._create_app(
            app_name=app_name,
            title="Contact Form",
            icon="",
            imports=imports,
            body=body,
            description="Fill out the form below.",
        )

    async def _create_app_from_task(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """Create app based on task analysis."""
        # Default to chat app
        return await self._create_chat_app(task, context)

    async def _edit_app(self, task: str, context: Dict[str, Any]) -> ProviderResult:
        """
        Edit an existing Streamlit app by adding features.

        Args:
            task: Natural language edit request (e.g., "add pdf download to arxiv app")
            context: Additional context

        Returns:
            ProviderResult with edit result
        """
        task_lower = task.lower()

        # Find which app to edit
        app_name = None
        for name in self._apps:
            if name in task_lower:
                app_name = name
                break

        # If no app found, try to find by domain keywords
        if not app_name:
            if any(kw in task_lower for kw in ['arxiv', 'paper']):
                app_name = 'arxiv_searcher'
            elif any(kw in task_lower for kw in ['stock', 'finance']):
                app_name = 'stock_dashboard'

        # Check if app exists on disk
        if app_name:
            app_path = self.workspace_dir / app_name / "app.py"
            if not app_path.exists():
                app_name = None

        if not app_name:
            # Try current app
            if self._current_app:
                app_name = self._current_app.name
            else:
                return ProviderResult(
                    success=False,
                    output=None,
                    error="No app found to edit. Create an app first or specify which app to modify.",
                    category=SkillCategory.APP_BUILDING,
                )

        app_path = self.workspace_dir / app_name / "app.py"

        # Find which feature to add
        feature_to_add = None
        for feature_name, feature_config in FEATURE_SNIPPETS.items():
            if any(kw in task_lower for kw in feature_config['keywords']):
                feature_to_add = feature_name
                break

        if not feature_to_add:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Could not determine which feature to add. Available features: {list(FEATURE_SNIPPETS.keys())}",
                category=SkillCategory.APP_BUILDING,
            )

        # Read existing app code
        try:
            existing_code = app_path.read_text()
        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Could not read app: {e}",
                category=SkillCategory.APP_BUILDING,
            )

        feature = FEATURE_SNIPPETS[feature_to_add]

        # Add import if needed
        if feature['import'] and feature['import'] not in existing_code:
            # Insert import after existing imports
            import_line = feature['import']
            if 'import streamlit as st' in existing_code:
                existing_code = existing_code.replace(
                    'import streamlit as st',
                    f'import streamlit as st\n{import_line}'
                )

        # Insert feature code after the marker
        insert_marker = feature['insert_after']
        if insert_marker in existing_code:
            # Find the line and insert after it
            lines = existing_code.split('\n')
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if insert_marker in line:
                    # Add the feature code
                    new_lines.append(feature['code'])

            existing_code = '\n'.join(new_lines)
        else:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Could not find insertion point '{insert_marker}' in app code.",
                category=SkillCategory.APP_BUILDING,
            )

        # Write updated code
        try:
            app_path.write_text(existing_code)
            logger.info(f"Added {feature_to_add} feature to {app_name}")

            return ProviderResult(
                success=True,
                output={
                    'app_name': app_name,
                    'feature_added': feature_to_add,
                    'app_path': str(app_path),
                    'message': f"Added {feature_to_add} to {app_name}",
                },
                category=SkillCategory.APP_BUILDING,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Failed to write updated app: {e}",
                category=SkillCategory.APP_BUILDING,
            )

    async def _create_app(
        self,
        app_name: str,
        title: str,
        icon: str = "",
        layout: str = "centered",
        imports: List[str] = None,
        body: str = "",
        description: str = "",
        extra_deps: List[str] = None,
    ) -> ProviderResult:
        """
        Create a Streamlit app.

        Args:
            app_name: Name of the app directory
            title: App title
            icon: Page icon emoji
            layout: Page layout (centered or wide)
            imports: Additional import statements
            body: Main app body code
            description: App description
            extra_deps: Additional pip dependencies

        Returns:
            ProviderResult with app info
        """
        try:
            # Create app directory
            app_path = self.workspace_dir / app_name
            app_path.mkdir(parents=True, exist_ok=True)

            # Build app code
            imports_str = '\n'.join(imports) if imports else ''
            description_block = f'st.markdown("{description}")' if description else ''

            code = APP_TEMPLATE.format(
                title=title,
                filename="app.py",
                icon=icon,
                layout=layout,
                imports=imports_str,
                description_block=description_block,
                body=body,
            )

            # Write app file
            app_file = app_path / "app.py"
            app_file.write_text(code)

            # Write requirements.txt if extra deps
            if extra_deps:
                deps = ["streamlit"] + extra_deps
                requirements_file = app_path / "requirements.txt"
                requirements_file.write_text('\n'.join(deps))

            # Create app object
            app = StreamlitApp(
                name=app_name,
                path=app_path,
                server_port=self._find_available_port(),
            )
            self._apps[app_name] = app
            self._current_app = app

            logger.info(f"Created Streamlit app: {app_name}")

            return ProviderResult(
                success=True,
                output={
                    'app_name': app_name,
                    'app_path': str(app_path),
                    'main_file': str(app_file),
                    'run_command': f"streamlit run {app_file}",
                    'title': title,
                    'message': f"App created! Run with: streamlit run {app_file}",
                },
                category=SkillCategory.APP_BUILDING,
                metadata={'extra_deps': extra_deps},
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Failed to create app: {e}",
                category=SkillCategory.APP_BUILDING,
            )

    async def _serve(self, context: Dict[str, Any] = None) -> ProviderResult:
        """Start the Streamlit dev server."""
        app = self._current_app
        if not app:
            return ProviderResult(
                success=False,
                output=None,
                error="No active app to serve. Create an app first.",
            )

        if app.is_running():
            return ProviderResult(
                success=True,
                output={
                    'message': 'Server already running',
                    'url': f"http://localhost:{app.server_port}",
                },
                category=SkillCategory.APP_BUILDING,
            )

        try:
            port = self._find_available_port(app.server_port)
            app.server_port = port

            app_file = app.path / "app.py"
            app.server_process = subprocess.Popen(
                ['streamlit', 'run', str(app_file), '--server.port', str(port), '--server.headless', 'true'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            await asyncio.sleep(3)

            if app.is_running():
                logger.info(f"Streamlit server started at http://localhost:{port}")
                return ProviderResult(
                    success=True,
                    output={
                        'message': 'Server started',
                        'url': f"http://localhost:{port}",
                        'app': app.name,
                    },
                    category=SkillCategory.APP_BUILDING,
                )
            else:
                return ProviderResult(
                    success=False,
                    output=None,
                    error="Server failed to start",
                )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Failed to start server: {e}",
            )

    async def _stop_server(self, context: Dict[str, Any] = None) -> ProviderResult:
        """Stop the Streamlit dev server."""
        app = self._current_app
        if not app or not app.is_running():
            return ProviderResult(
                success=True,
                output={'message': 'No server running'},
                category=SkillCategory.APP_BUILDING,
            )

        try:
            app.server_process.terminate()
            app.server_process.wait(timeout=5)
            app.server_process = None

            logger.info("Streamlit server stopped")

            return ProviderResult(
                success=True,
                output={'message': 'Server stopped'},
                category=SkillCategory.APP_BUILDING,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Failed to stop server: {e}",
            )

    def _find_available_port(self, start_port: int = None) -> int:
        """Find an available port."""
        import socket

        port = start_port or self.default_port
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1

        return port

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        return self.is_available and self.is_initialized

    def get_app(self, name: str) -> Optional[StreamlitApp]:
        """Get an app by name."""
        return self._apps.get(name)

    def list_apps(self) -> List[str]:
        """List all known apps."""
        return list(self._apps.keys())

    async def cleanup(self):
        """Clean up resources."""
        for app in self._apps.values():
            if app.is_running():
                try:
                    app.server_process.terminate()
                    app.server_process.wait(timeout=5)
                except Exception:
                    pass

        self._apps.clear()
        self._current_app = None


# Factory function for registration
def create_provider(config: Dict[str, Any] = None) -> StreamlitProvider:
    """Create and return a StreamlitProvider instance."""
    return StreamlitProvider(config)
