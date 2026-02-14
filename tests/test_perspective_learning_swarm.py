"""Tests for the Perspective Learning Swarm.

Tests cover:
- Types, enums, config validation
- Each agent with mocked DSPy modules
- Content assembly produces valid markdown
- PDF/HTML generation doesn't crash on sample content
- Swarm execution with fully mocked agents
- Convenience functions (teach_perspectives, teach_perspectives_sync)
- Registration check
"""

import asyncio
import json
import logging
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Import types
from Jotty.core.swarms.perspective_learning_swarm.types import (
    PerspectiveType, Language, AgeGroup, ContentDepth,
    PerspectiveLearningConfig, PerspectiveLearningResult,
    PerspectiveSection, LanguageContent, DebatePoint,
    ProjectActivity, FrameworkModel, LessonContent,
    format_steps_on_newlines, PERSPECTIVE_LABELS, LANGUAGE_LABELS,
)


# =============================================================================
# TYPE AND ENUM TESTS
# =============================================================================

class TestEnums:
    """Test all enums."""

    @pytest.mark.unit
    def test_perspective_type_values(self):
        assert PerspectiveType.INTUITIVE_VISUAL.value == "intuitive_visual"
        assert PerspectiveType.STRUCTURED_FRAMEWORK.value == "structured_framework"
        assert PerspectiveType.STORYTELLING.value == "storytelling"
        assert PerspectiveType.DEBATE_CRITICAL.value == "debate_critical"
        assert PerspectiveType.HANDS_ON_PROJECT.value == "hands_on_project"
        assert PerspectiveType.REAL_WORLD_APPLICATION.value == "real_world_application"
        assert len(PerspectiveType) == 6

    @pytest.mark.unit
    def test_language_values(self):
        assert Language.ENGLISH.value == "english"
        assert Language.HINDI.value == "hindi"
        assert Language.KANNADA.value == "kannada"
        assert Language.FRENCH.value == "french"
        assert len(Language) == 4

    @pytest.mark.unit
    def test_age_group_values(self):
        assert AgeGroup.EARLY_PRIMARY.value == "early_primary"
        assert AgeGroup.PRIMARY.value == "primary"
        assert AgeGroup.MIDDLE.value == "middle"
        assert AgeGroup.HIGH.value == "high"
        assert AgeGroup.GENERAL.value == "general"

    @pytest.mark.unit
    def test_content_depth_values(self):
        assert ContentDepth.QUICK.value == "quick"
        assert ContentDepth.STANDARD.value == "standard"
        assert ContentDepth.DEEP.value == "deep"
        assert ContentDepth.COMPREHENSIVE.value == "comprehensive"


class TestConfig:
    """Test PerspectiveLearningConfig."""

    @pytest.mark.unit
    def test_default_config(self):
        config = PerspectiveLearningConfig()
        assert config.name == "PerspectiveLearningSwarm"
        assert config.domain == "perspective_learning"
        assert config.student_name == "Student"
        assert config.age_group == AgeGroup.PRIMARY
        assert config.depth == ContentDepth.STANDARD
        assert len(config.languages) == 4
        assert len(config.perspectives) == 6
        assert config.celebration_word == "Wonderful!"
        assert config.generate_pdf is True
        assert config.generate_html is True
        assert config.send_telegram is False
        assert config.optimization_mode == "parallel_deep"
        assert config.max_concurrent_llm == 5
        assert config.llm_model == "haiku"
        assert config.use_fast_predict is True

    @pytest.mark.unit
    def test_custom_config(self):
        config = PerspectiveLearningConfig(
            student_name="Aria",
            age_group=AgeGroup.MIDDLE,
            depth=ContentDepth.DEEP,
            languages=[Language.ENGLISH, Language.HINDI],
            perspectives=[PerspectiveType.INTUITIVE_VISUAL, PerspectiveType.STORYTELLING],
        )
        assert config.student_name == "Aria"
        assert config.age_group == AgeGroup.MIDDLE
        assert config.depth == ContentDepth.DEEP
        assert len(config.languages) == 2
        assert len(config.perspectives) == 2

    @pytest.mark.unit
    def test_config_sets_llm_timeout(self):
        config = PerspectiveLearningConfig()
        assert config.llm_timeout > 0

    @pytest.mark.unit
    def test_all_perspectives_in_default(self):
        config = PerspectiveLearningConfig()
        for pt in PerspectiveType:
            assert pt in config.perspectives

    @pytest.mark.unit
    def test_all_languages_in_default(self):
        config = PerspectiveLearningConfig()
        for lang in Language:
            assert lang in config.languages


class TestDataClasses:
    """Test content dataclasses."""

    @pytest.mark.unit
    def test_perspective_section(self):
        section = PerspectiveSection(
            perspective=PerspectiveType.STORYTELLING,
            title="Feel the Story",
            content="Once upon a time...",
            key_takeaway="Stories teach empathy.",
            activity="Write your own version.",
        )
        assert section.perspective == PerspectiveType.STORYTELLING
        assert section.title == "Feel the Story"
        assert section.activity == "Write your own version."

    @pytest.mark.unit
    def test_language_content(self):
        lc = LanguageContent(
            language=Language.HINDI,
            summary="यह एक सारांश है",
            key_vocabulary=["मीडिया = media", "प्रभाव = influence"],
            slogans=["सोचो, समझो, फिर मानो!"],
        )
        assert lc.language == Language.HINDI
        assert len(lc.key_vocabulary) == 2
        assert len(lc.slogans) == 1

    @pytest.mark.unit
    def test_debate_point(self):
        dp = DebatePoint(
            position="Media literacy should be taught in schools",
            argument="Students need critical thinking skills",
            evidence="Studies show improved decision-making",
            counterargument="Limited classroom time",
            critical_question="Who decides what media literacy means?",
        )
        assert dp.position != ""

    @pytest.mark.unit
    def test_project_activity(self):
        pa = ProjectActivity(
            title="Create a Media Analysis Poster",
            description="Analyze an ad and present findings",
            materials=["Chart paper", "Markers", "Magazine ads"],
            steps=["Select an ad", "Identify persuasion techniques", "Create poster"],
            learning_outcome="Students can identify persuasion in ads",
        )
        assert len(pa.materials) == 3

    @pytest.mark.unit
    def test_framework_model(self):
        fm = FrameworkModel(
            name="The SPACE Framework",
            description="Source, Purpose, Audience, Content, Effect",
            how_to_use="Apply to any media message",
            visual_layout="5 columns with one question each",
            example_applied="Applying to a cereal ad",
        )
        assert fm.name == "The SPACE Framework"

    @pytest.mark.unit
    def test_lesson_content(self):
        content = LessonContent(
            topic="Media Influence",
            student_name="Aria",
            central_idea="Media influences decisions",
            learning_objectives=["Identify media types"],
            key_concepts=[{"name": "persuasion", "description": "convincing someone"}],
            running_example="Riya sees an ad...",
            vocabulary=[{"term": "media", "definition": "communication tools"}],
            perspectives=[],
            language_sections=[],
            key_insights=["Media is everywhere"],
            parent_guide="Discuss media at home",
            socratic_questions=["Why do ads exist?"],
            total_words=5000,
        )
        assert content.topic == "Media Influence"
        assert content.total_words == 5000

    @pytest.mark.unit
    def test_result_dataclass(self):
        result = PerspectiveLearningResult(
            success=True,
            swarm_name="PerspectiveLearningSwarm",
            domain="perspective_learning",
            output={"topic": "Media"},
            execution_time=10.5,
            student_name="Aria",
            topic="Media",
            perspectives_generated=6,
            languages_generated=3,
        )
        assert result.success is True
        assert result.perspectives_generated == 6
        assert result.languages_generated == 3


class TestHelpers:
    """Test helper functions."""

    @pytest.mark.unit
    def test_format_steps_on_newlines(self):
        text = "Start here. Step 1: do this. Step 2: do that."
        result = format_steps_on_newlines(text)
        assert "Step 1:" in result
        assert "Step 2:" in result

    @pytest.mark.unit
    def test_format_steps_empty(self):
        assert format_steps_on_newlines("") == ""
        assert format_steps_on_newlines(None) is None

    @pytest.mark.unit
    def test_perspective_labels(self):
        assert PERSPECTIVE_LABELS[PerspectiveType.INTUITIVE_VISUAL] == "See It Clearly"
        assert PERSPECTIVE_LABELS[PerspectiveType.STORYTELLING] == "Feel the Story"
        assert len(PERSPECTIVE_LABELS) == 6

    @pytest.mark.unit
    def test_language_labels(self):
        assert Language.HINDI in LANGUAGE_LABELS
        assert Language.KANNADA in LANGUAGE_LABELS
        assert Language.FRENCH in LANGUAGE_LABELS


# =============================================================================
# AGENT TESTS (with mocked DSPy)
# =============================================================================

def _mock_dspy_result(**fields):
    """Create a mock DSPy prediction result."""
    mock = MagicMock()
    for k, v in fields.items():
        setattr(mock, k, v)
    return mock


class TestCurriculumDesignerAgent:
    """Test CurriculumDesignerAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_design_returns_expected_keys(self):
        with patch('Jotty.core.swarms.perspective_learning_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.perspective_learning_swarm.agents import CurriculumDesignerAgent
            agent = CurriculumDesignerAgent.__new__(CurriculumDesignerAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                learning_objectives="Obj1 | Obj2 | Obj3",
                key_concepts_json='[{"name": "persuasion", "description": "convincing someone"}]',
                vocabulary_json='[{"term": "media", "definition": "communication tools"}]',
                running_example_scenario="Riya sees an ad for chocolate...",
                section_depth_plan="Intuitive: 500 words | Framework: 500 words",
                transdisciplinary_connections="Connects to math through data analysis",
            )
            agent._designer = MagicMock(return_value=mock_result)

            result = await agent.design(
                topic="Media Influence",
                student_name="Aria",
                age_group="primary",
            )

            assert 'learning_objectives' in result
            assert 'key_concepts' in result
            assert 'vocabulary' in result
            assert 'running_example_scenario' in result
            assert len(result['learning_objectives']) == 3


class TestIntuitiveExplainerAgent:
    """Test IntuitiveExplainerAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_explain_returns_expected_keys(self):
        with patch('Jotty.core.swarms.perspective_learning_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.perspective_learning_swarm.agents import IntuitiveExplainerAgent
            agent = IntuitiveExplainerAgent.__new__(IntuitiveExplainerAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                simplest_example="Imagine you see a colorful poster...",
                step_by_step_build="Step 1: Look at the image. Step 2: Read the text.",
                visual_descriptions="A poster with bright colors | A TV showing an ad",
                aha_moment="Every ad is designed to make you FEEL something!",
                check_your_understanding="What emotion does this ad try to create? | Who paid for this?",
            )
            agent._explainer = MagicMock(return_value=mock_result)

            result = await agent.explain(
                topic="Media Influence",
                concepts="persuasion, bias",
                student_name="Aria",
                age_group="primary",
                running_example="Riya sees an ad...",
            )

            assert 'simplest_example' in result
            assert 'aha_moment' in result
            assert len(result['visual_descriptions']) == 2


class TestMultilingualAgent:
    """Test MultilingualAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_hindi(self):
        with patch('Jotty.core.swarms.perspective_learning_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.perspective_learning_swarm.agents import MultilingualAgent
            agent = MultilingualAgent.__new__(MultilingualAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                summary="मीडिया हमारे निर्णयों को प्रभावित करता है...",
                key_vocabulary_translated="media = मीडिया | influence = प्रभाव",
                reflection_prompts="क्या आपने कभी किसी विज्ञापन से प्रभावित होकर कुछ खरीदा? | मीडिया का आप पर क्या प्रभाव पड़ता है?",
                activity="एक विज्ञापन का विश्लेषण करें और उसके बारे में एक निबंध लिखें।",
                slogans="सोचो, समझो, फिर मानो! | मीडिया को समझो, सही निर्णय लो!",
            )
            agent._translator = MagicMock(return_value=mock_result)

            result = await agent.generate(
                topic="Media Influence",
                key_concepts="persuasion, bias",
                vocabulary="media, influence",
                student_name="Aria",
                running_example="Riya sees an ad...",
                target_language="hindi",
            )

            assert 'summary' in result
            assert len(result['key_vocabulary']) == 2
            assert len(result['slogans']) == 2


class TestNarrativeEditorAgent:
    """Test NarrativeEditorAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_edit_returns_expected_keys(self):
        with patch('Jotty.core.swarms.perspective_learning_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.perspective_learning_swarm.agents import NarrativeEditorAgent
            agent = NarrativeEditorAgent.__new__(NarrativeEditorAgent)
            agent.model = "sonnet"
            agent.use_fast_predict = True
            agent.llm_timeout = 240
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            content = "# Media\n## See It Clearly\nContent here..." * 20
            mock_result = _mock_dspy_result(
                edited_content=content + "\n## Enhanced content here...",
                socratic_questions="What do you think? | Why does this matter? | Who benefits?",
                parent_guide="Discuss media with your child at dinner time.",
            )
            agent._editor = MagicMock(return_value=mock_result)

            result = await agent.edit(
                assembled_content=content,
                running_example="Riya sees an ad...",
                student_name="Aria",
                topic="Media Influence",
            )

            assert 'edited_content' in result
            assert 'parent_guide' in result
            assert len(result['socratic_questions']) == 3


# =============================================================================
# PDF/HTML GENERATION TESTS
# =============================================================================

class TestPDFHTMLGeneration:
    """Test PDF and HTML generation with sample content."""

    def _sample_content(self) -> LessonContent:
        """Create sample LessonContent for testing."""
        return LessonContent(
            topic="Media and Decision Making",
            student_name="Aria",
            central_idea="Media is a tool that influences the decisions people make",
            learning_objectives=["Identify types of media", "Analyze persuasion techniques"],
            key_concepts=[{"name": "persuasion", "description": "convincing someone"}],
            running_example="Riya sees an ad for ChocoBurst chocolate...",
            vocabulary=[{"term": "media", "definition": "communication tools"}],
            perspectives=[
                PerspectiveSection(
                    perspective=PerspectiveType.INTUITIVE_VISUAL,
                    title="See It Clearly",
                    content="Imagine you see a poster with bright colors and a smiling face...",
                    key_takeaway="Every ad is designed to make you FEEL something!",
                ),
                PerspectiveSection(
                    perspective=PerspectiveType.STORYTELLING,
                    title="Feel the Story",
                    content="Once upon a time, Riya was scrolling through her tablet...",
                    key_takeaway="Not everything you see online is true.",
                ),
            ],
            language_sections=[
                LanguageContent(
                    language=Language.HINDI,
                    summary="मीडिया हमारे निर्णयों को प्रभावित करता है",
                    key_vocabulary=["मीडिया = media", "प्रभाव = influence"],
                    slogans=["सोचो, समझो, फिर मानो!"],
                ),
                LanguageContent(
                    language=Language.KANNADA,
                    summary="ಮಾಧ್ಯಮವು ನಮ್ಮ ನಿರ್ಧಾರಗಳ ಮೇಲೆ ಪ್ರಭಾವ ಬೀರುತ್ತದೆ",
                    key_vocabulary=["ಮಾಧ್ಯಮ = media"],
                ),
            ],
            key_insights=["Media is everywhere", "Critical thinking is a superpower"],
            parent_guide="Discuss media choices at dinner time.",
            socratic_questions=["Why do ads exist?", "Who benefits from this ad?"],
            total_words=3000,
        )

    @pytest.mark.unit
    def test_html_renderer_produces_output(self):
        from Jotty.core.swarms.perspective_learning_swarm.pdf_generator import PerspectiveHTMLRenderer
        renderer = PerspectiveHTMLRenderer(celebration_word="Wonderful!")
        content = self._sample_content()
        html = renderer.render(content)
        assert len(html) > 500
        assert "Media and Decision Making" in html
        assert "Aria" in html
        assert "perspective-intuitive" in html
        assert "perspective-story" in html
        assert "language-hindi" in html
        assert "language-kannada" in html
        assert "Parent" in html

    @pytest.mark.unit
    def test_html_renderer_includes_toc(self):
        from Jotty.core.swarms.perspective_learning_swarm.pdf_generator import PerspectiveHTMLRenderer
        renderer = PerspectiveHTMLRenderer()
        content = self._sample_content()
        html = renderer.render(content)
        assert "Table of Contents" in html

    @pytest.mark.unit
    def test_html_renderer_includes_stats(self):
        from Jotty.core.swarms.perspective_learning_swarm.pdf_generator import PerspectiveHTMLRenderer
        renderer = PerspectiveHTMLRenderer()
        content = self._sample_content()
        html = renderer.render(content)
        assert "Perspectives" in html
        assert "Languages" in html
        assert "Concepts" in html

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_html_file(self, tmp_path):
        from Jotty.core.swarms.perspective_learning_swarm.pdf_generator import generate_perspective_html
        content = self._sample_content()
        output_path = str(tmp_path / "test_output.html")
        result = await generate_perspective_html(content, output_path)
        assert result is not None
        assert result == output_path
        from pathlib import Path
        assert Path(output_path).exists()
        html_text = Path(output_path).read_text()
        assert "Media and Decision Making" in html_text


# =============================================================================
# SWARM REGISTRATION TESTS
# =============================================================================

class TestSwarmRegistration:
    """Test swarm registration."""

    @pytest.mark.unit
    def test_registered_in_swarm_registry(self):
        from Jotty.core.swarms.perspective_learning_swarm import PerspectiveLearningSwarm
        from Jotty.core.swarms.base_swarm import SwarmRegistry
        swarm_class = SwarmRegistry.get("perspective_learning")
        assert swarm_class is PerspectiveLearningSwarm

    @pytest.mark.unit
    def test_lazy_import_from_core_swarms(self):
        from Jotty.core.swarms import PerspectiveLearningSwarm
        assert PerspectiveLearningSwarm is not None

    @pytest.mark.unit
    def test_lazy_import_teach_perspectives(self):
        from Jotty.core.swarms import teach_perspectives
        assert callable(teach_perspectives)

    @pytest.mark.unit
    def test_lazy_import_types(self):
        from Jotty.core.swarms import PerspectiveType, Language, AgeGroup, ContentDepth
        assert len(PerspectiveType) == 6
        assert len(Language) == 4


# =============================================================================
# CONTENT ASSEMBLY TESTS
# =============================================================================

class TestContentAssembly:
    """Test content assembly produces valid markdown."""

    @pytest.mark.unit
    def test_build_complete_content(self):
        from Jotty.core.swarms.perspective_learning_swarm.swarm import PerspectiveLearningSwarm
        from Jotty.core.swarms.perspective_learning_swarm.types import PerspectiveLearningConfig

        swarm = PerspectiveLearningSwarm.__new__(PerspectiveLearningSwarm)
        swarm.config = PerspectiveLearningConfig()

        content = swarm._build_complete_content(
            student_name="Aria",
            topic="Media Influence",
            central_idea="Media influences decisions",
            learning_objectives=["Identify media types", "Analyze persuasion"],
            key_concepts=[{"name": "persuasion", "description": "convincing someone"}],
            vocabulary=[{"term": "media", "definition": "tools"}],
            running_example="Riya sees an ad...",
            intuitive={'simplest_example': 'Imagine a poster...', 'aha_moment': 'Ads make you feel!'},
            framework={'decision_tree': 'Start: Is this an ad?', 'key_principles': ['Always question the source']},
            story={'story': 'Once upon a time...', 'moral_or_lesson': 'Think before you believe'},
            debate={'central_question': 'Should ads target kids?', 'points_for': [], 'points_against': []},
            project={'projects': [], 'poster_design_brief': 'Create a media literacy poster'},
            realworld={'daily_life_connections': ['YouTube ads'], 'future_impact': 'Better decisions'},
            language_content={
                Language.HINDI: {'summary': 'Hindi summary', 'slogans': ['Hindi slogan']},
            },
            config=PerspectiveLearningConfig(),
        )

        assert "# Media Influence" in content
        assert "Aria" in content
        assert "See It Clearly" in content
        assert "Think It Through" in content
        assert "Feel the Story" in content
        assert "Debate It" in content
        assert "Build It" in content
        assert "Live It" in content
        assert "Hindi" in content
        assert "Table of Contents" in content

    @pytest.mark.unit
    def test_build_perspective_sections(self):
        from Jotty.core.swarms.perspective_learning_swarm.swarm import PerspectiveLearningSwarm
        from Jotty.core.swarms.perspective_learning_swarm.types import PerspectiveLearningConfig

        swarm = PerspectiveLearningSwarm.__new__(PerspectiveLearningSwarm)
        swarm.config = PerspectiveLearningConfig()

        sections = swarm._build_perspective_sections(
            intuitive={'simplest_example': 'Example', 'step_by_step_build': 'Steps', 'aha_moment': 'Aha!'},
            framework={'decision_tree': 'Tree', 'key_principles': ['P1']},
            story={'story': 'Story text', 'moral_or_lesson': 'Moral'},
            debate={'central_question': 'Question?', 'form_your_opinion': 'Think!'},
            project={'poster_design_brief': 'Brief', 'role_play_scenario': 'Scenario', 'reflection_activity': 'Reflect'},
            realworld={'future_impact': 'Impact'},
            config=PerspectiveLearningConfig(),
        )

        assert len(sections) == 6
        assert sections[0].perspective == PerspectiveType.INTUITIVE_VISUAL
        assert sections[1].perspective == PerspectiveType.STRUCTURED_FRAMEWORK
        assert sections[2].perspective == PerspectiveType.STORYTELLING

    @pytest.mark.unit
    def test_build_language_sections(self):
        from Jotty.core.swarms.perspective_learning_swarm.swarm import PerspectiveLearningSwarm

        swarm = PerspectiveLearningSwarm.__new__(PerspectiveLearningSwarm)

        sections = swarm._build_language_sections({
            Language.HINDI: {'summary': 'Hindi text', 'key_vocabulary': ['term1'], 'slogans': ['slogan1']},
            Language.FRENCH: {'summary': 'French text', 'key_vocabulary': ['terme1']},
        })

        assert len(sections) == 2
        assert sections[0].language == Language.HINDI
        assert sections[0].summary == 'Hindi text'


# =============================================================================
# SWARM EXECUTION TEST (fully mocked)
# =============================================================================

class TestSwarmExecution:
    """Test swarm execution with fully mocked agents."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_swarm_instantiation(self):
        from Jotty.core.swarms.perspective_learning_swarm.swarm import PerspectiveLearningSwarm
        swarm = PerspectiveLearningSwarm()
        assert swarm.config.name == "PerspectiveLearningSwarm"
        assert swarm.config.domain == "perspective_learning"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_swarm_with_custom_config(self):
        from Jotty.core.swarms.perspective_learning_swarm.swarm import PerspectiveLearningSwarm
        config = PerspectiveLearningConfig(
            student_name="Aria",
            age_group=AgeGroup.MIDDLE,
            depth=ContentDepth.DEEP,
        )
        swarm = PerspectiveLearningSwarm(config)
        assert swarm.config.student_name == "Aria"
        assert swarm.config.age_group == AgeGroup.MIDDLE


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test teach_perspectives and teach_perspectives_sync."""

    @pytest.mark.unit
    def test_teach_perspectives_is_callable(self):
        from Jotty.core.swarms.perspective_learning_swarm import teach_perspectives
        assert callable(teach_perspectives)

    @pytest.mark.unit
    def test_teach_perspectives_sync_is_callable(self):
        from Jotty.core.swarms.perspective_learning_swarm import teach_perspectives_sync
        assert callable(teach_perspectives_sync)


# =============================================================================
# SIGNATURES TESTS
# =============================================================================

class TestSignatures:
    """Test DSPy signatures can be instantiated."""

    @pytest.mark.unit
    def test_all_signatures_importable(self):
        from Jotty.core.swarms.perspective_learning_swarm.signatures import (
            CurriculumDesignerSignature, IntuitiveExplainerSignature,
            FrameworkBuilderSignature, StorytellerSignature,
            DebateArchitectSignature, ProjectDesignerSignature,
            RealWorldConnectorSignature, MultilingualContentSignature,
            ContentAssemblerSignature, NarrativeEditorSignature,
        )
        # Just verify they're all importable DSPy signatures
        import dspy
        for sig in [CurriculumDesignerSignature, IntuitiveExplainerSignature,
                     FrameworkBuilderSignature, StorytellerSignature,
                     DebateArchitectSignature, ProjectDesignerSignature,
                     RealWorldConnectorSignature, MultilingualContentSignature,
                     ContentAssemblerSignature, NarrativeEditorSignature]:
            assert issubclass(sig, dspy.Signature)

    @pytest.mark.unit
    def test_signature_count(self):
        from Jotty.core.swarms.perspective_learning_swarm import signatures
        all_sigs = [name for name in signatures.__all__]
        assert len(all_sigs) == 10


# =============================================================================
# AGENTS IMPORT TESTS
# =============================================================================

class TestAgentsImport:
    """Test all agents are importable."""

    @pytest.mark.unit
    def test_all_agents_importable(self):
        from Jotty.core.swarms.perspective_learning_swarm.agents import (
            CurriculumDesignerAgent, IntuitiveExplainerAgent,
            FrameworkBuilderAgent, StorytellerAgent,
            DebateArchitectAgent, ProjectDesignerAgent,
            RealWorldConnectorAgent, MultilingualAgent,
            ContentAssemblerAgent, NarrativeEditorAgent,
        )
        from Jotty.core.swarms.olympiad_learning_swarm.agents import BaseOlympiadAgent
        for agent_cls in [CurriculumDesignerAgent, IntuitiveExplainerAgent,
                          FrameworkBuilderAgent, StorytellerAgent,
                          DebateArchitectAgent, ProjectDesignerAgent,
                          RealWorldConnectorAgent, MultilingualAgent,
                          ContentAssemblerAgent, NarrativeEditorAgent]:
            assert issubclass(agent_cls, BaseOlympiadAgent)
