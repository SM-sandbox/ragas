"""
Prompt Manager - Jinja2-based template rendering for prompts.

Centralizes all prompt template loading and rendering for the ragas evaluation system.
Templates are stored as .jinja2 files in config/prompts/.
"""

import logging
import pathlib
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates using Jinja2.

    Prompts are stored as .jinja2 files in config/prompts/.
    Supports subdirectories for organization (e.g., judge/, generation/).
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize PromptManager with a prompts directory.

        Args:
            prompts_dir: Path to prompts directory. Defaults to config/prompts/
                        relative to repo root.
        """
        if prompts_dir is None:
            # Default to config/prompts relative to repo root
            # lib/core/prompt_manager.py -> go up 3 levels to repo root
            repo_root = pathlib.Path(__file__).parent.parent.parent
            prompts_dir = repo_root / "config" / "prompts"

        self.prompts_dir = pathlib.Path(prompts_dir)

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

        logger.info(f"PromptManager initialized with prompts_dir={self.prompts_dir}")

    def render(self, template_path: str, **variables: Any) -> str:
        """
        Render a prompt template with variables.

        Args:
            template_path: Path to template relative to prompts_dir
                          (e.g., "judge/answer_evaluation.jinja2")
            **variables: Template variables to substitute

        Returns:
            Rendered prompt string

        Raises:
            TemplateNotFound: If template file doesn't exist
            jinja2.TemplateError: If template has syntax errors
        """
        try:
            template = self.env.get_template(template_path)
            rendered = template.render(**variables)
            return rendered
        except TemplateNotFound:
            logger.error(f"Template not found: {template_path}")
            raise
        except Exception as e:
            logger.error(f"Error rendering template {template_path}: {e}")
            raise

    def list_templates(self, subdirectory: str = "") -> list[str]:
        """
        List available templates in a subdirectory.

        Args:
            subdirectory: Subdirectory to list (e.g., "judge", "generation")

        Returns:
            List of template filenames
        """
        search_dir = self.prompts_dir / subdirectory
        if not search_dir.exists():
            return []
        return [f.name for f in search_dir.glob("*.jinja2")]

    def template_exists(self, template_path: str) -> bool:
        """Check if a template exists."""
        full_path = self.prompts_dir / template_path
        return full_path.exists()


# Singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    Get the singleton PromptManager instance.

    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def reset_prompt_manager() -> None:
    """
    Reset the singleton instance. Useful for testing.
    """
    global _prompt_manager
    _prompt_manager = None
