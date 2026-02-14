"""
LOTUS-powered ArXiv Operations
==============================

Uses LOTUS semantic operators for enhanced paper discovery, filtering, and analysis.
Provides intelligent paper ranking, concept extraction, and multi-paper summarization.

Features:
- Semantic paper search and ranking
- Intelligent filtering by relevance, difficulty, prerequisites
- Structured concept extraction from abstracts
- Cross-paper insight aggregation
- Full-text extraction for deep analysis

Usage:
    from Jotty.core.skills.research.lotus_arxiv import LotusArxiv

    lotus_arxiv = LotusArxiv()

    # Search and rank papers semantically
    papers = await lotus_arxiv.search_and_rank(
        query="transformer attention mechanism",
        rank_by="most exciting and impactful for learning",
        limit=5
    )

    # Extract concepts from paper
    concepts = await lotus_arxiv.extract_concepts(paper_id="1706.03762")

    # Summarize insights across papers
    summary = await lotus_arxiv.aggregate_insights(paper_ids=["1706.03762", "2301.07041"])
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import pandas for type hints (always available)
import pandas as pd

# Check LOTUS availability
try:
    import lotus
    from lotus import WebSearchCorpus, web_search
    from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
    from lotus.vector_store import FaissVS

    # web_extract may not be available in all LOTUS versions
    try:
        from lotus import web_extract
        WEB_EXTRACT_AVAILABLE = True
    except ImportError:
        WEB_EXTRACT_AVAILABLE = False
        web_extract = None

    LOTUS_AVAILABLE = True
except ImportError:
    LOTUS_AVAILABLE = False
    WEB_EXTRACT_AVAILABLE = False
    web_extract = None
    logger.warning("LOTUS not available. Install with: pip install lotus-ai")


@dataclass
class LotusArxivConfig:
    """Configuration for LOTUS ArXiv operations."""
    model: str = "gpt-4o-mini"  # LLM for semantic operations
    embedding_model: str = "intfloat/e5-base-v2"  # Embedding model
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"  # Reranker
    use_reranker: bool = True
    use_cot: bool = True  # Chain-of-thought for better reasoning
    max_search_results: int = 100  # Initial search pool


class LotusArxiv:
    """
    LOTUS-powered ArXiv operations for enhanced paper discovery and analysis.

    Provides semantic search, ranking, filtering, concept extraction, and
    cross-paper summarization using LOTUS operators.
    """

    def __init__(self, config: Optional[LotusArxivConfig] = None):
        self.config = config or LotusArxivConfig()
        self._initialized = False

        if not LOTUS_AVAILABLE:
            raise ImportError("LOTUS not available. Install with: pip install lotus-ai")

    def _init_lotus(self) -> None:
        """Initialize LOTUS with configured models."""
        if self._initialized:
            return

        logger.info(" Initializing LOTUS...")

        # Language model for semantic operations
        self.lm = LM(model=self.config.model)

        # Retrieval model for embeddings
        self.rm = SentenceTransformersRM(model=self.config.embedding_model)

        # Configure LOTUS
        if self.config.use_reranker:
            self.reranker = CrossEncoderReranker(model=self.config.reranker_model)
            self.vs = FaissVS()
            lotus.settings.configure(
                lm=self.lm,
                rm=self.rm,
                reranker=self.reranker,
                vs=self.vs
            )
        else:
            lotus.settings.configure(lm=self.lm, rm=self.rm)

        self._initialized = True
        logger.info(" LOTUS initialized")

    async def search_papers(
        self,
        query: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Search ArXiv papers using LOTUS.

        Args:
            query: Search query
            limit: Maximum number of papers to return

        Returns:
            DataFrame with columns: title, abstract, authors, arxiv_id, url, etc.
        """
        self._init_lotus()

        logger.info(f" Searching ArXiv: {query}")

        # Run in executor since LOTUS web_search may block
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None,
            lambda: web_search(WebSearchCorpus.ARXIV, query, limit)
        )

        logger.info(f"  Found {len(df)} papers")
        return df

    async def search_and_rank(
        self,
        query: str,
        rank_by: str = "Which {abstract} is most exciting and impactful for learning?",
        limit: int = 10,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Search ArXiv and rank papers semantically.

        Args:
            query: Search query
            rank_by: Natural language ranking criterion (use {column} for column references)
            limit: Initial search pool size
            top_k: Number of top papers to return

        Returns:
            DataFrame with top papers ranked by criterion
        """
        self._init_lotus()

        logger.info(f" Searching and ranking: {query}")

        loop = asyncio.get_running_loop()

        # Search
        df = await loop.run_in_executor(
            None,
            lambda: web_search(WebSearchCorpus.ARXIV, query, limit)
        )

        if df.empty:
            logger.warning("No papers found")
            return df

        # Rank using sem_topk
        strategy = "cot" if self.config.use_cot else "quick"
        ranked_df = await loop.run_in_executor(
            None,
            lambda: df.sem_topk(rank_by, K=top_k, method="heap")[0]
        )

        logger.info(f"  Ranked top {len(ranked_df)} papers")
        return ranked_df

    async def filter_papers(
        self,
        df: pd.DataFrame,
        filter_by: str
    ) -> pd.DataFrame:
        """
        Filter papers using natural language predicate.

        Args:
            df: DataFrame of papers
            filter_by: Natural language filter (e.g., "{abstract} is about deep learning")

        Returns:
            Filtered DataFrame
        """
        self._init_lotus()

        logger.info(f" Filtering papers: {filter_by}")

        loop = asyncio.get_running_loop()
        strategy = "cot" if self.config.use_cot else "quick"

        filtered_df = await loop.run_in_executor(
            None,
            lambda: df.sem_filter(filter_by, strategy=strategy)
        )

        logger.info(f"  {len(filtered_df)} papers match filter")
        return filtered_df

    async def extract_full_text(self, arxiv_id: str) -> Optional[str]:
        """
        Extract full text from an ArXiv paper.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "1706.03762")

        Returns:
            Full paper text, or None if extraction failed
        """
        if not WEB_EXTRACT_AVAILABLE:
            logger.warning("web_extract not available in this LOTUS version")
            return None

        self._init_lotus()

        logger.info(f" Extracting full text: {arxiv_id}")

        loop = asyncio.get_running_loop()

        try:
            df = await loop.run_in_executor(
                None,
                lambda: web_extract(WebSearchCorpus.ARXIV, doc_id=arxiv_id)
            )

            if df is not None and "full_text" in df.columns and len(df) > 0:
                text = df["full_text"].iloc[0]
                logger.info(f"  Extracted {len(text)} characters")
                return text
            return None
        except Exception as e:
            logger.error(f"Full text extraction failed: {e}")
            return None

    async def extract_concepts(
        self,
        text: str,
        output_fields: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Extract structured concepts from text using sem_extract.

        Args:
            text: Paper abstract or full text
            output_fields: Dict mapping output column names to descriptions
                          Default: extracts name, description, difficulty, prerequisites

        Returns:
            DataFrame with extracted concepts
        """
        self._init_lotus()

        if output_fields is None:
            output_fields = {
                "concept_name": "The name of a key concept or technique",
                "description": "A brief description of what it does",
                "difficulty": "Difficulty level: beginner, intermediate, or advanced",
                "prerequisites": "What you need to know first"
            }

        logger.info(" Extracting concepts...")

        loop = asyncio.get_running_loop()

        # Create a DataFrame with the text
        df = pd.DataFrame({"text": [text]})

        # Extract using sem_extract
        result_df = await loop.run_in_executor(
            None,
            lambda: df.sem_extract(["text"], output_fields)
        )

        logger.info(f"  Extracted {len(result_df)} concepts")
        return result_df

    async def aggregate_insights(
        self,
        df: pd.DataFrame,
        aggregation: str = "Summarize the key innovations and insights from all {abstract}"
    ) -> str:
        """
        Aggregate insights across multiple papers using sem_agg.

        Args:
            df: DataFrame of papers with abstract column
            aggregation: Natural language aggregation instruction

        Returns:
            Aggregated summary string
        """
        self._init_lotus()

        logger.info(" Aggregating insights across papers...")

        loop = asyncio.get_running_loop()

        result_df = await loop.run_in_executor(
            None,
            lambda: df.sem_agg(aggregation)
        )

        summary = result_df._output[0] if hasattr(result_df, '_output') else str(result_df)
        logger.info(f"  Generated {len(summary)} char summary")
        return summary

    async def find_similar_papers(
        self,
        paper_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        similarity_column: str = "abstract",
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find papers similar to given papers using semantic similarity.

        Args:
            paper_df: DataFrame with paper(s) to find similar to
            corpus_df: DataFrame of papers to search within
            similarity_column: Column to use for similarity matching
            top_k: Number of similar papers to return

        Returns:
            DataFrame of similar papers
        """
        self._init_lotus()

        logger.info(f" Finding similar papers...")

        loop = asyncio.get_running_loop()

        # Index the corpus
        indexed_df = await loop.run_in_executor(
            None,
            lambda: corpus_df.sem_index(similarity_column, "temp_index")
        )

        # Search for similar
        similar_df = await loop.run_in_executor(
            None,
            lambda: indexed_df.sem_search(
                similarity_column,
                paper_df[similarity_column].iloc[0],
                K=top_k
            )
        )

        logger.info(f"  Found {len(similar_df)} similar papers")
        return similar_df

    async def map_papers(
        self,
        df: pd.DataFrame,
        instruction: str
    ) -> pd.DataFrame:
        """
        Transform each paper using natural language instruction.

        Args:
            df: DataFrame of papers
            instruction: Natural language transformation (e.g., "Explain why {title} matters in 2 sentences")

        Returns:
            DataFrame with new column containing transformed values
        """
        self._init_lotus()

        logger.info(f" Mapping papers: {instruction}")

        loop = asyncio.get_running_loop()

        result_df = await loop.run_in_executor(
            None,
            lambda: df.sem_map(instruction)
        )

        return result_df


def is_lotus_available() -> bool:
    """Check if LOTUS is available."""
    return LOTUS_AVAILABLE


__all__ = ['LotusArxiv', 'LotusArxivConfig', 'is_lotus_available']
