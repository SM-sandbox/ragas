"""
Unit tests for PromptManager in ragas.

Tests cover:
- PromptManager class functionality
- Template loading and rendering
- Judge and generation templates
"""

import pytest
from jinja2 import TemplateNotFound


class TestPromptManager:
    """Test PromptManager class core functionality."""

    @pytest.fixture
    def temp_prompts_dir(self, tmp_path):
        """Create a temporary prompts directory with test templates."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Create subdirectories
        (prompts_dir / "judge").mkdir()
        (prompts_dir / "generation").mkdir()

        # Create test templates
        (prompts_dir / "simple.jinja2").write_text("Hello {{ name }}!")
        (prompts_dir / "judge" / "test.jinja2").write_text(
            "Question: {{ question }}\nAnswer: {{ answer }}"
        )

        return prompts_dir

    @pytest.fixture
    def pm(self, temp_prompts_dir):
        """Create PromptManager with temp directory."""
        from lib.core.prompt_manager import PromptManager
        return PromptManager(prompts_dir=str(temp_prompts_dir))

    def test_init_with_valid_directory(self, temp_prompts_dir):
        """PromptManager initializes with valid directory."""
        from lib.core.prompt_manager import PromptManager
        pm = PromptManager(prompts_dir=str(temp_prompts_dir))
        assert pm.prompts_dir == temp_prompts_dir

    def test_init_with_invalid_directory_raises(self, tmp_path):
        """PromptManager raises FileNotFoundError for invalid directory."""
        from lib.core.prompt_manager import PromptManager
        with pytest.raises(FileNotFoundError):
            PromptManager(prompts_dir=str(tmp_path / "nonexistent"))

    def test_render_simple_variable(self, pm):
        """Basic variable substitution works."""
        result = pm.render("simple.jinja2", name="World")
        assert result == "Hello World!"

    def test_render_multiple_variables(self, pm):
        """Multiple variables are substituted correctly."""
        result = pm.render("judge/test.jinja2", question="What is X?", answer="X is Y")
        assert "Question: What is X?" in result
        assert "Answer: X is Y" in result

    def test_render_missing_template_raises(self, pm):
        """Missing template file raises TemplateNotFound."""
        with pytest.raises(TemplateNotFound):
            pm.render("nonexistent.jinja2", name="test")

    def test_list_templates(self, pm):
        """list_templates returns available templates."""
        templates = pm.list_templates("judge")
        assert "test.jinja2" in templates

    def test_template_exists_true(self, pm):
        """template_exists returns True for existing template."""
        assert pm.template_exists("simple.jinja2") is True

    def test_template_exists_false(self, pm):
        """template_exists returns False for nonexistent template."""
        assert pm.template_exists("nonexistent.jinja2") is False


class TestPromptManagerSingleton:
    """Test singleton behavior of get_prompt_manager."""

    def test_singleton_returns_same_instance(self):
        """get_prompt_manager() returns the same instance."""
        from lib.core.prompt_manager import get_prompt_manager, reset_prompt_manager
        reset_prompt_manager()

        pm1 = get_prompt_manager()
        pm2 = get_prompt_manager()
        assert pm1 is pm2

        reset_prompt_manager()

    def test_reset_clears_singleton(self):
        """reset_prompt_manager clears the singleton."""
        from lib.core.prompt_manager import get_prompt_manager, reset_prompt_manager
        reset_prompt_manager()

        pm1 = get_prompt_manager()
        reset_prompt_manager()
        pm2 = get_prompt_manager()

        assert pm1 is not pm2
        reset_prompt_manager()


class TestJudgeTemplates:
    """Test judge templates with real templates."""

    @pytest.fixture
    def pm(self):
        """Get the real PromptManager with actual templates."""
        from lib.core.prompt_manager import get_prompt_manager, reset_prompt_manager
        reset_prompt_manager()
        return get_prompt_manager()

    def test_answer_evaluation_template_renders(self, pm):
        """Answer evaluation template renders without error."""
        result = pm.render(
            "judge/answer_evaluation.jinja2",
            question="What is the voltage?",
            ground_truth="480V",
            answer="The voltage is 480V.",
            context="[1] manual.pdf: Voltage is 480V",
        )
        assert "What is the voltage?" in result
        assert "480V" in result
        assert "correctness" in result.lower()
        assert "verdict" in result.lower()

    def test_judge_template_contains_scoring_criteria(self, pm):
        """Judge template includes all scoring criteria."""
        result = pm.render(
            "judge/answer_evaluation.jinja2",
            question="test",
            ground_truth="test",
            answer="test",
            context="test",
        )
        assert "correctness" in result.lower()
        assert "completeness" in result.lower()
        assert "faithfulness" in result.lower()
        assert "relevance" in result.lower()
        assert "clarity" in result.lower()


class TestGenerationTemplates:
    """Test question generation templates."""

    @pytest.fixture
    def pm(self):
        """Get the real PromptManager with actual templates."""
        from lib.core.prompt_manager import get_prompt_manager, reset_prompt_manager
        reset_prompt_manager()
        return get_prompt_manager()

    def test_single_hop_template_renders(self, pm):
        """Single-hop generation template renders."""
        result = pm.render(
            "generation/single_hop.jinja2",
            doc_title="Test Document",
            source_filename="test.pdf",
            content="This is test content.",
            difficulty="medium",
        )
        assert "Test Document" in result
        assert "test.pdf" in result
        assert "medium" in result.lower()

    def test_multi_hop_template_renders(self, pm):
        """Multi-hop generation template renders."""
        result = pm.render(
            "generation/multi_hop.jinja2",
            doc1_title="Document 1",
            content1="Content from doc 1",
            doc2_title="Document 2",
            content2="Content from doc 2",
            difficulty="hard",
        )
        assert "Document 1" in result
        assert "Document 2" in result
        assert "hard" in result.lower()

    def test_relevance_eval_template_renders(self, pm):
        """Relevance evaluation template renders."""
        result = pm.render(
            "generation/relevance_eval.jinja2",
            question="What is the voltage?",
            answer="480V",
            source="manual.pdf",
        )
        assert "What is the voltage?" in result
        assert "480V" in result
        assert "manual.pdf" in result


class TestEvaluatorIntegration:
    """Test that evaluator uses Jinja2 templates correctly."""

    def test_judge_prompt_uses_template(self):
        """GoldEvaluator._judge_answer uses Jinja2 template."""
        from lib.core.prompt_manager import get_prompt_manager, reset_prompt_manager

        reset_prompt_manager()
        pm = get_prompt_manager()

        # Verify template renders correctly
        result = pm.render(
            "judge/answer_evaluation.jinja2",
            question="What is the max current?",
            ground_truth="100A",
            answer="The maximum current is 100A.",
            context="[1] spec.pdf: Max current is 100A",
        )

        assert "What is the max current?" in result
        assert "100A" in result
        assert "JSON" in result  # Should ask for JSON response

        reset_prompt_manager()
