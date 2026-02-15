"""ArXiv Learning Swarm - Agent implementations."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy

from Jotty.core.modes.agent.base import BaseSwarmAgent, DomainAgent, DomainAgentConfig

from .signatures import (
    ConceptExtractionSignature,
    ContentPolisherSignature,
    ExampleGeneratorSignature,
    IntuitionBuilderSignature,
    MathSimplifierSignature,
    ProgressiveBuilderSignature,
    SingleConceptDeepSignature,
    UnifiedConceptLearningSignature,
)
from .types import (
    AudienceLevel,
    Concept,
    ContentStyle,
    LearningContent,
    LearningDepth,
    LearningSection,
    PaperInfo,
    format_steps_on_newlines,
)

# Import LOTUS
try:
    from Jotty.core.capabilities.skills.research.lotus_arxiv import (
        LotusArxiv,
        LotusArxivConfig,
        is_lotus_available,
    )

    LOTUS_AVAILABLE = is_lotus_available()
except ImportError:
    LOTUS_AVAILABLE = False
    LotusArxiv = None
    LotusArxivConfig = None

logger = logging.getLogger(__name__)


class BaseSwarmAgent(BaseSwarmAgent):
    """Base class for arxiv learning agents. Extends BaseSwarmAgent with LLM model selection."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 90,
    ) -> None:
        super().__init__(
            memory=memory, context=context, bus=bus, learned_context=learned_context, signature=None
        )
        self.model = model
        self.use_fast_predict = use_fast_predict
        self.llm_timeout = llm_timeout
        self._lm = None

    def _get_lm(self) -> Any:
        """Get or create LLM instance. Tries Direct API first, then CLI fallback."""
        if self._lm is None:
            # If already configured globally, reuse it
            if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
                self._lm = dspy.settings.lm
                return self._lm

            # Try direct Anthropic API first (fastest, no subprocess)
            try:
                from Jotty.core.infrastructure.foundation.direct_anthropic_lm import (
                    DirectAnthropicLM,
                    is_api_key_available,
                )

                if is_api_key_available():
                    self._lm = DirectAnthropicLM(model=self.model, max_tokens=8192)
                    dspy.configure(lm=self._lm)
                    return self._lm
            except Exception as e:
                logger.debug(f"DirectAnthropicLM not available: {e}")

            # Fallback to Claude CLI
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI

                self._lm = DirectClaudeCLI(model=self.model)
                dspy.configure(lm=self._lm)
            except Exception as e:
                logger.warning(f"Could not init LLM: {e}")
        return self._lm

    def _create_module(self, signature: Any) -> Any:
        """Create dspy module - Predict (fast) or ChainOfThought (reasoning)."""
        self._get_lm()
        if self.use_fast_predict:
            return dspy.Predict(signature)
        else:
            return dspy.ChainOfThought(signature)


class PaperFetcherAgent(BaseSwarmAgent):
    """Fetches papers from ArXiv via the ``arxiv-downloader`` skill.

    All networking, caching, retry, and XML parsing is delegated to the
    skill layer.  The only agent-specific logic retained is LOTUS
    semantic ranking (``search_and_rank_with_lotus``).
    """

    # ---- helpers to call skill ----

    def _get_skill_tools(self) -> Any:
        """Lazy-load the arxiv-downloader skill tools."""
        if not hasattr(self, "_skill_tools"):
            try:
                from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

                registry = get_skills_registry()
                registry.init()
                skill = registry.get_skill("arxiv-downloader")
                self._skill_tools = skill.tools if skill else {}
            except Exception:
                self._skill_tools = {}
        return self._skill_tools

    @staticmethod
    def _result_to_paper(d: Dict[str, Any]) -> PaperInfo:
        """Convert a skill result dict to a PaperInfo dataclass."""
        return PaperInfo(
            arxiv_id=d.get("arxiv_id") or d.get("id", ""),
            title=d.get("title", ""),
            authors=d.get("authors", []),
            abstract=d.get("abstract", ""),
            categories=d.get("categories", []),
            published=d.get("published", ""),
            pdf_url=d.get("pdf_url", ""),
            arxiv_url=d.get("arxiv_url") or d.get("url", ""),
        )

    # ---- public interface (unchanged) ----

    async def fetch_by_id(self, arxiv_id: str) -> Optional[PaperInfo]:
        """Fetch paper by ArXiv ID with caching (delegated to skill)."""
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]

        tools = self._get_skill_tools()
        download_fn = tools.get("download_arxiv_paper_tool")
        if download_fn is None:
            logger.warning("arxiv-downloader skill not available, cannot fetch paper")
            return None

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, download_fn, {"arxiv_id": arxiv_id, "extract_mode": "text"}
            )
            if result.get("success"):
                paper = self._result_to_paper(result)
                self._broadcast(
                    "paper_fetched", {"arxiv_id": paper.arxiv_id, "title": paper.title[:50]}
                )
                return paper
            return None
        except Exception as e:
            logger.error(f"Paper fetch failed: {e}")
            return None

    async def search_by_topic(self, topic: str, max_results: int = 5) -> List[PaperInfo]:
        """Search papers by topic (delegated to skill)."""
        tools = self._get_skill_tools()
        search_fn = tools.get("search_arxiv_tool")
        if search_fn is None:
            logger.warning("arxiv-downloader skill not available, cannot search")
            return []

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                search_fn,
                {"query": f"all:{topic}", "max_results": max_results, "sort_by": "relevance"},
            )
            if result.get("success"):
                papers = [self._result_to_paper(r) for r in result.get("results", [])]
                self._broadcast("papers_searched", {"count": len(papers)})
                return papers
            return []
        except Exception as e:
            logger.error(f"Topic search failed: {e}")
            return []

    # ---- LOTUS (agent-specific, not a skill) ----

    async def search_and_rank_with_lotus(
        self,
        topic: str,
        max_results: int = 5,
        rank_by: str = "Which {abstract} is most exciting and impactful for learning?",
        lotus_model: str = "gpt-4o-mini",
    ) -> List[PaperInfo]:
        """Search papers using LOTUS with semantic ranking.

        Falls back to ``search_by_topic`` when LOTUS is unavailable.
        """
        if not LOTUS_AVAILABLE:
            logger.warning("LOTUS not available, falling back to standard search")
            return await self.search_by_topic(topic, max_results)

        try:
            logger.info(f"LOTUS-powered search: {topic}")
            lotus_arxiv = LotusArxiv(LotusArxivConfig(model=lotus_model))

            ranked_df = await lotus_arxiv.search_and_rank(
                query=topic,
                rank_by=rank_by,
                limit=max_results * 3,
                top_k=max_results,
            )

            if ranked_df.empty:
                logger.warning("LOTUS search returned no results")
                return await self.search_by_topic(topic, max_results)

            papers = []
            for _, row in ranked_df.iterrows():
                arxiv_id = ""
                if "arxiv_id" in row:
                    arxiv_id = str(row["arxiv_id"])
                elif "url" in row:
                    url = str(row["url"])
                    if "/abs/" in url:
                        arxiv_id = url.split("/abs/")[-1]
                    elif "/pdf/" in url:
                        arxiv_id = url.split("/pdf/")[-1].replace(".pdf", "")

                papers.append(
                    PaperInfo(
                        arxiv_id=arxiv_id,
                        title=str(row.get("title", "")),
                        authors=(
                            row.get("authors", []) if isinstance(row.get("authors"), list) else []
                        ),
                        abstract=str(row.get("abstract", "")),
                        categories=[],
                        published=(
                            str(row.get("published", ""))[:10] if row.get("published") else ""
                        ),
                        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
                        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                    )
                )

            logger.info(f"  Found {len(papers)} semantically ranked papers")
            return papers
        except Exception as e:
            logger.error(f"LOTUS search failed: {e}, falling back to standard search")
            return await self.search_by_topic(topic, max_results)

        return papers


class ConceptExtractorAgent(BaseSwarmAgent):
    """Extracts concepts from papers."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._extractor = self._create_module(ConceptExtractionSignature)

    async def extract(self, paper: PaperInfo) -> List[Concept]:
        """Extract concepts from paper."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._extractor(
                paper_title=paper.title,
                abstract=paper.abstract,
                full_text_summary=f"Based on abstract only{context_suffix}",
            )

            try:
                # Strip markdown code blocks if present
                concepts_json = str(result.concepts)
                if concepts_json.startswith("```"):
                    # Remove ```json and ``` markers
                    lines = concepts_json.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    concepts_json = "\n".join(lines)
                concepts_data = json.loads(concepts_json)
            except Exception as e:
                logger.warning(f"Failed to parse concepts JSON: {e}")
                concepts_data = []

            concepts = []
            for c in concepts_data:
                # Ensure prerequisites is a list, not a string
                prereqs = c.get("prerequisites", [])
                if isinstance(prereqs, str):
                    # Convert string to list (split by comma or just wrap it)
                    prereqs = (
                        [p.strip() for p in prereqs.split(",") if p.strip()]
                        if "," in prereqs
                        else [prereqs]
                    )
                elif not isinstance(prereqs, list):
                    prereqs = []

                concepts.append(
                    Concept(
                        name=c.get("name", ""),
                        description=c.get("description", ""),
                        why_it_matters=c.get("why_it_matters", ""),
                        prerequisites=prereqs,
                        difficulty=int(c.get("difficulty", 3)),
                        math_required=c.get("math_required", True),
                    )
                )

            self._broadcast("concepts_extracted", {"count": len(concepts)})

            return concepts

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []


class IntuitionBuilderAgent(BaseSwarmAgent):
    """Builds intuition for concepts."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._builder = self._create_module(IntuitionBuilderSignature)

    async def build(self, concept: Concept, audience_level: str) -> Dict[str, Any]:
        """Build intuition for a concept."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._builder(
                concept=f"{concept.name}: {concept.description}{context_suffix}",
                why_it_matters=concept.why_it_matters,
                audience_level=audience_level,
                prerequisites=(
                    ", ".join(concept.prerequisites) if concept.prerequisites else "Basic math"
                ),
            )

            self._broadcast("intuition_built", {"concept": concept.name})

            return {
                "hook": str(result.hook),
                "analogy": str(result.analogy),
                "intuition_build": str(result.intuition_build),
                "aha_moment": str(result.aha_moment),
                "check_understanding": str(result.check_understanding),
            }

        except Exception as e:
            logger.error(f"Intuition building failed: {e}")
            return {}


class MathSimplifierAgent(BaseSwarmAgent):
    """Simplifies math to be accessible."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._simplifier = self._create_module(MathSimplifierSignature)

    async def simplify(
        self, concept: Concept, intuition: Dict[str, Any], audience_level: str
    ) -> Dict[str, Any]:
        """Simplify math for a concept."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._simplifier(
                concept=f"{concept.name}: {concept.description}{context_suffix}",
                intuition=intuition.get("intuition_build", ""),
                equations="Key equations from the paper",
                audience_level=audience_level,
            )

            self._broadcast("math_simplified", {"concept": concept.name})

            return {
                "math_motivation": str(result.math_motivation),
                "building_blocks": str(result.building_blocks),
                "step_by_step": str(result.step_by_step),
                "concrete_example": str(result.concrete_example),
                "connection_to_intuition": str(result.connection_to_intuition),
            }

        except Exception as e:
            logger.error(f"Math simplification failed: {e}")
            return {}


class ExampleGeneratorAgent(BaseSwarmAgent):
    """Generates examples to reinforce learning."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._generator = self._create_module(ExampleGeneratorSignature)

    async def generate(
        self, concept: Concept, intuition: Dict[str, Any], math: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate examples for a concept."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._generator(
                concept=f"{concept.name}: {concept.description}{context_suffix}",
                intuition=intuition.get("intuition_build", ""),
                math_explanation=math.get("step_by_step", ""),
            )

            what_ifs = [w.strip() for w in str(result.what_if_variations).split("|") if w.strip()]

            self._broadcast("examples_generated", {"concept": concept.name})

            return {
                "simple_example": str(result.simple_example),
                "intermediate_example": str(result.intermediate_example),
                "challenging_example": str(result.challenging_example),
                "code_example": str(result.code_example),
                "what_if_variations": what_ifs,
            }

        except Exception as e:
            logger.error(f"Example generation failed: {e}")
            return {}


class ProgressiveBuilderAgent(BaseSwarmAgent):
    """Builds progressive learning content."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._builder = self._create_module(ProgressiveBuilderSignature)

    async def build(
        self,
        paper: PaperInfo,
        concepts: List[Concept],
        intuitions: Dict[str, Dict],
        math_explanations: Dict[str, Dict],
        examples: Dict[str, Dict],
        celebration_word: str,
    ) -> Dict[str, Any]:
        """Build complete progressive learning content."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._builder(
                paper_info=json.dumps(
                    {
                        "title": paper.title,
                        "abstract": paper.abstract[:500] + context_suffix,
                        "authors": paper.authors[:3],
                    }
                ),
                concepts=json.dumps(
                    [{"name": c.name, "description": c.description} for c in concepts]
                ),
                intuitions=json.dumps(intuitions),
                math_explanations=json.dumps(math_explanations),
                examples=json.dumps(examples),
                celebration_word=celebration_word,
            )

            key_insights = [k.strip() for k in str(result.key_insights).split("|") if k.strip()]
            next_steps = [n.strip() for n in str(result.next_steps).split("|") if n.strip()]

            self._broadcast("content_built", {"paper": paper.title[:30]})

            return {
                "complete_content": str(result.complete_content),
                "key_insights": key_insights,
                "summary": str(result.summary),
                "next_steps": next_steps,
            }

        except Exception as e:
            logger.error(f"Progressive building failed: {e}")
            return {}


class ContentPolisherAgent(BaseSwarmAgent):
    """Polishes content to be engaging."""

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._polisher = self._create_module(ContentPolisherSignature)

    async def polish(self, draft_content: str, style: str, audience: str) -> Dict[str, Any]:
        """Polish content for engagement."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._polisher(
                draft_content=draft_content, style=f"{style}{context_suffix}", audience=audience
            )

            self._broadcast(
                "content_polished",
                {"engagement": float(result.engagement_score) if result.engagement_score else 0},
            )

            return {
                "polished_content": str(result.polished_content),
                "engagement_score": (
                    float(result.engagement_score) if result.engagement_score else 75.0
                ),
                "clarity_score": float(result.clarity_score) if result.clarity_score else 75.0,
            }

        except Exception as e:
            logger.error(f"Content polishing failed: {e}")
            return {
                "polished_content": draft_content,
                "engagement_score": 50.0,
                "clarity_score": 50.0,
            }


class UnifiedLearningAgent(BaseSwarmAgent):
    """
    Optimized learning content generator with two modes:

    1. generate_all() - Single LLM call for all concepts (faster, less detailed)
    2. generate_parallel() - Parallel LLM calls per concept (slower, MUCH more detailed)

    For 30+ page documents, use generate_parallel() which runs concept generation
    concurrently for both speed AND quality.
    """

    # Cache for generated content (paper_id → content)
    _content_cache: Dict[str, Dict] = {}

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        model: str = "haiku",
        use_fast_predict: bool = True,
        llm_timeout: int = 60,
    ) -> None:
        super().__init__(
            memory, context, bus, learned_context, model, use_fast_predict, llm_timeout
        )
        self._generator = self._create_module(UnifiedConceptLearningSignature)
        self._deep_generator = self._create_module(SingleConceptDeepSignature)

    async def generate_parallel(
        self,
        paper: "PaperInfo",
        concepts: List["Concept"],
        audience_level: str,
        celebration_word: str = "Bingo!",
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate DEEP content for each concept IN PARALLEL.

        This produces 30+ page documents by running multiple LLM calls concurrently.
        Each concept gets thorough treatment with detailed intuition, math, and examples.
        """
        cache_key = f"parallel_{paper.arxiv_id}_{audience_level}_{len(concepts)}"
        if cache_key in self._content_cache:
            logger.info(f" Using cached parallel content for {paper.arxiv_id}")
            return self._content_cache[cache_key]

        paper_context = f"{paper.title}: {paper.abstract[:300]}"

        # Semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_one_concept(concept: "Concept") -> Dict[str, Any]:
            """Generate deep content for a single concept."""
            async with semaphore:
                try:
                    result = self._deep_generator(
                        concept_name=concept.name,
                        concept_description=concept.description,
                        why_it_matters=concept.why_it_matters,
                        paper_context=paper_context,
                        audience_level=audience_level,
                    )
                    return {
                        "name": concept.name,
                        "analogy": str(result.analogy),
                        "intuition": str(result.intuition),
                        "aha_moment": str(result.aha_moment),
                        "math_motivation": str(result.math_motivation),
                        "math_steps": str(result.math_steps),
                        "simple_example": str(result.simple_example),
                        "advanced_example": str(result.advanced_example),
                        "code_example": str(result.code_example),
                    }
                except Exception as e:
                    logger.error(f"Failed to generate content for {concept.name}: {e}")
                    return {
                        "name": concept.name,
                        "analogy": "",
                        "intuition": concept.description,
                        "aha_moment": concept.why_it_matters,
                        "math_motivation": "",
                        "math_steps": "",
                        "simple_example": "",
                        "advanced_example": "",
                        "code_example": "",
                    }

        # Run all concept generations in parallel
        logger.info(
            f" Generating deep content for {len(concepts)} concepts in parallel (max {max_concurrent} concurrent)..."
        )
        start = datetime.now()

        tasks = [generate_one_concept(c) for c in concepts]
        concept_results = await asyncio.gather(*tasks)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f" Parallel generation complete in {elapsed:.1f}s")

        # Build result structure
        intuitions = {}
        math_explanations = {}
        examples = {}

        for cr in concept_results:
            name = cr["name"]
            intuitions[name] = {
                "hook": f"Let's understand {name}",
                "analogy": cr["analogy"],
                "intuition_build": cr["intuition"],
                "aha_moment": cr["aha_moment"],
                "check_understanding": "",
            }
            math_explanations[name] = {
                "math_motivation": cr["math_motivation"],
                "building_blocks": "",
                "step_by_step": cr["math_steps"],
                "concrete_example": cr["simple_example"],
                "connection_to_intuition": "",
            }
            examples[name] = {
                "simple_example": cr["simple_example"],
                "intermediate_example": cr["advanced_example"],
                "challenging_example": "",
                "code_example": cr["code_example"],
                "what_if_variations": [],
            }

        result_data = {
            "hook": f"Let's explore the revolutionary ideas in {paper.title}",
            "concepts": concept_results,
            "key_insights": [
                f"{celebration_word}! {cr['aha_moment'][:100]}"
                for cr in concept_results
                if cr["aha_moment"]
            ],
            "summary": f"We explored {len(concepts)} key concepts from {paper.title}",
            "next_steps": [
                "Implement these concepts in code",
                "Read related papers",
                "Apply to your own projects",
            ],
            "intuitions": intuitions,
            "math_explanations": math_explanations,
            "examples": examples,
            "complete_content": self._build_deep_content(concept_results, paper, celebration_word),
        }

        self._content_cache[cache_key] = result_data
        return result_data

    def _build_deep_content(
        self, concept_results: List[Dict], paper: "PaperInfo", celebration_word: str
    ) -> str:
        """Build comprehensive markdown content from parallel results."""
        sections = [f"# {paper.title}\n\n"]

        for cr in concept_results:
            sections.append(f"\n## {cr['name']}\n")

            if cr.get("analogy"):
                sections.append(f"\n**Think of it like this:** {cr['analogy']}\n")

            if cr.get("intuition"):
                sections.append(f"\n### Building Intuition\n\n{cr['intuition']}\n")

            if cr.get("aha_moment"):
                sections.append(f"\n **{celebration_word}!** {cr['aha_moment']}\n")

            if cr.get("math_motivation") or cr.get("math_steps"):
                sections.append(f"\n### The Mathematics\n")
                if cr.get("math_motivation"):
                    sections.append(f"\n{cr['math_motivation']}\n")
                if cr.get("math_steps"):
                    sections.append(f"\n{cr['math_steps']}\n")

            if cr.get("simple_example"):
                sections.append(f"\n### Example\n\n{cr['simple_example']}\n")

            if cr.get("advanced_example"):
                sections.append(f"\n### Advanced Example\n\n{cr['advanced_example']}\n")

            if cr.get("code_example"):
                sections.append(
                    f"\n### Code Implementation\n\n```python\n{cr['code_example']}\n```\n"
                )

        return "\n".join(sections)

    async def generate_all(
        self,
        paper: "PaperInfo",
        concepts: List["Concept"],
        audience_level: str,
        celebration_word: str = "Bingo!",
    ) -> Dict[str, Any]:
        """
        Generate complete learning content for all concepts in ONE LLM call.

        Returns dict with:
        - hook: Opening hook
        - concepts: List of concept learning content
        - key_insights: List of key insights
        - summary: Summary text
        - next_steps: List of next steps
        - intuitions: Dict for backward compatibility
        - math_explanations: Dict for backward compatibility
        - examples: Dict for backward compatibility
        """
        # Check cache first
        cache_key = f"{paper.arxiv_id}_{audience_level}"
        if cache_key in self._content_cache:
            logger.info(f" Using cached learning content for {paper.arxiv_id}")
            return self._content_cache[cache_key]

        try:
            # FULL context for quality content
            concepts_data = [
                {
                    "name": c.name,
                    "description": c.description,
                    "why_it_matters": c.why_it_matters,
                    "difficulty": c.difficulty,
                    "math_required": c.math_required,
                }
                for c in concepts[:7]  # Up to 7 concepts for comprehensive coverage
            ]

            # Full context for high-quality generation
            result = self._generator(
                paper_title=paper.title,
                paper_abstract=paper.abstract,
                concepts_json=json.dumps(concepts_data),
                audience_level=audience_level,
                celebration_word=celebration_word,
            )

            # Parse the JSON output
            try:
                content = json.loads(str(result.learning_content_json))
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                raw = str(result.learning_content_json)
                # Find JSON in response
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        content = json.loads(raw[start:end])
                    except Exception:
                        content = self._fallback_content(paper, concepts, celebration_word)
                else:
                    content = self._fallback_content(paper, concepts, celebration_word)

            # Build backward-compatible structures
            intuitions = {}
            math_explanations = {}
            examples = {}

            for concept_data in content.get("concepts", []):
                name = concept_data.get("name", "")
                if not name:
                    continue

                intuitions[name] = {
                    "hook": content.get("hook", ""),
                    "analogy": concept_data.get("analogy", ""),
                    "intuition_build": concept_data.get("intuition", ""),
                    "aha_moment": concept_data.get("aha_moment", ""),
                    "check_understanding": "",
                }

                math_explanations[name] = {
                    "math_motivation": concept_data.get("math_motivation", ""),
                    "building_blocks": "",
                    "step_by_step": concept_data.get("math_steps", ""),
                    "concrete_example": concept_data.get("simple_example", ""),
                    "connection_to_intuition": "",
                }

                examples[name] = {
                    "simple_example": concept_data.get("simple_example", ""),
                    "intermediate_example": "",
                    "challenging_example": "",
                    "code_example": concept_data.get("code_example", ""),
                    "what_if_variations": [],
                }

            result_data = {
                "hook": content.get("hook", f"Let's understand {paper.title}"),
                "concepts": content.get("concepts", []),
                "key_insights": content.get("key_insights", []),
                "summary": content.get("summary", ""),
                "next_steps": content.get("next_steps", []),
                # Backward compatibility
                "intuitions": intuitions,
                "math_explanations": math_explanations,
                "examples": examples,
                "complete_content": self._build_complete_content(content, paper),
            }

            # Cache the result
            self._content_cache[cache_key] = result_data

            self._broadcast(
                "unified_learning_complete",
                {
                    "paper": paper.title[:30],
                    "concepts_count": len(content.get("concepts", [])),
                    "insights_count": len(content.get("key_insights", [])),
                },
            )

            return result_data

        except Exception as e:
            logger.error(f"Unified learning generation failed: {e}")
            import traceback

            traceback.print_exc()
            return self._fallback_content(paper, concepts, celebration_word)

    def _fallback_content(
        self, paper: "PaperInfo", concepts: List["Concept"], celebration_word: str
    ) -> Dict:
        """Generate minimal fallback content if LLM fails."""
        return {
            "hook": f"Let's explore {paper.title}",
            "concepts": [
                {"name": c.name, "analogy": "", "intuition": c.description} for c in concepts[:3]
            ],
            "key_insights": [f"Understanding {c.name} is key" for c in concepts[:3]],
            "summary": f"This paper introduces {concepts[0].name if concepts else 'new ideas'}",
            "next_steps": ["Explore related papers", "Try implementing the concepts"],
            "intuitions": {},
            "math_explanations": {},
            "examples": {},
            "complete_content": paper.abstract,
        }

    def _build_complete_content(self, content: Dict, paper: "PaperInfo") -> str:
        """Build complete markdown content from structured data."""
        sections = []

        # Hook
        sections.append(f"# {paper.title}\n\n{content.get('hook', '')}")

        # Concepts
        for c in content.get("concepts", []):
            name = c.get("name", "Concept")
            sections.append(f"\n## {name}\n")

            if c.get("analogy"):
                sections.append(f"**Analogy:** {c['analogy']}\n")

            if c.get("intuition"):
                sections.append(f"\n### Understanding {name}\n{c['intuition']}\n")

            if c.get("aha_moment"):
                sections.append(
                    f"\n **{content.get('celebration_word', 'Bingo!')}!** {c['aha_moment']}\n"
                )

            if c.get("math_steps"):
                sections.append(
                    f"\n### The Math\n{c.get('math_motivation', '')}\n\n{c['math_steps']}\n"
                )

            if c.get("code_example"):
                sections.append(f"\n### Code Example\n```python\n{c['code_example']}\n```\n")

        # Summary
        if content.get("summary"):
            sections.append(f"\n## Summary\n{content['summary']}\n")

        # Key Insights
        if content.get("key_insights"):
            sections.append("\n## Key Insights\n")
            for insight in content["key_insights"]:
                sections.append(f"- {insight}\n")

        # Next Steps
        if content.get("next_steps"):
            sections.append("\n## What's Next?\n")
            for step in content["next_steps"]:
                sections.append(f"- {step}\n")

        return "\n".join(sections)


# =============================================================================
# ARXIV LEARNING SWARM
# =============================================================================
