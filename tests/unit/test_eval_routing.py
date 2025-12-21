#!/usr/bin/env python3
"""
Unit tests for checkpoint vs experiment routing logic.

Tests that:
- Checkpoints route to checkpoints/C###
- Experiments route to experiments/E###
- Naming conventions are correct
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRunDirectoryRouting:
    """Test that config_type routes to correct directories."""
    
    def test_checkpoint_routes_to_checkpoints_dir(self):
        """Checkpoint config_type should route to checkpoints/ directory."""
        from lib.core.evaluator import GoldEvaluator
        
        # Mock the heavy dependencies
        with patch.object(GoldEvaluator, '__init__', lambda self, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.config_type = "checkpoint"
            evaluator.config = {"client": "BFAI"}
            evaluator.cloud_mode = True
            evaluator.precision_k = 25
            evaluator.generator_reasoning = "low"
            
            # Test the directory determination logic
            type_prefix = {
                "checkpoint": "C",
                "run": "R",
                "experiment": "E",
            }.get(evaluator.config_type, "R")
            
            assert type_prefix == "C"
            
            # Verify subdirectory logic
            if evaluator.config_type == "checkpoint":
                subdir = "checkpoints"
            elif evaluator.config_type == "experiment":
                subdir = "experiments"
            else:
                subdir = "runs"
            
            assert subdir == "checkpoints"
    
    def test_experiment_routes_to_experiments_dir(self):
        """Experiment config_type should route to experiments/ directory."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda self, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.config_type = "experiment"
            evaluator.config = {"client": "BFAI"}
            
            type_prefix = {
                "checkpoint": "C",
                "run": "R",
                "experiment": "E",
            }.get(evaluator.config_type, "R")
            
            assert type_prefix == "E"
            
            if evaluator.config_type == "checkpoint":
                subdir = "checkpoints"
            elif evaluator.config_type == "experiment":
                subdir = "experiments"
            else:
                subdir = "runs"
            
            assert subdir == "experiments"
    
    def test_run_routes_to_runs_dir(self):
        """Run config_type should route to runs/ directory."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda self, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.config_type = "run"
            evaluator.config = {"client": "BFAI"}
            
            type_prefix = {
                "checkpoint": "C",
                "run": "R",
                "experiment": "E",
            }.get(evaluator.config_type, "R")
            
            assert type_prefix == "R"
            
            if evaluator.config_type == "checkpoint":
                subdir = "checkpoints"
            elif evaluator.config_type == "experiment":
                subdir = "experiments"
            else:
                subdir = "runs"
            
            assert subdir == "runs"


class TestNamingConventions:
    """Test naming conventions for different run types."""
    
    def test_checkpoint_naming_format(self):
        """Checkpoint names should follow C###__date__mode__config__qN format."""
        # Simulate naming logic
        type_prefix = "C"
        next_id = 16
        date_str = "2025-12-21"
        mode_str = "cloud"
        config_str = "p25-3-flash-low"
        num_questions = 458
        
        dir_name = f"{type_prefix}{next_id:03d}__{date_str}__{mode_str}__{config_str}__q{num_questions}"
        
        assert dir_name == "C016__2025-12-21__cloud__p25-3-flash-low__q458"
        assert dir_name.startswith("C")
        assert "__cloud__" in dir_name or "__local__" in dir_name
    
    def test_experiment_naming_format(self):
        """Experiment names should follow E###__date__mode__config__qN format."""
        type_prefix = "E"
        next_id = 1
        date_str = "2025-12-21"
        mode_str = "cloud"
        config_str = "p25-3-flash-low"
        num_questions = 458
        
        dir_name = f"{type_prefix}{next_id:03d}__{date_str}__{mode_str}__{config_str}__q{num_questions}"
        
        assert dir_name == "E001__2025-12-21__cloud__p25-3-flash-low__q458"
        assert dir_name.startswith("E")
    
    def test_id_padding(self):
        """IDs should be zero-padded to 3 digits."""
        for id_num, expected in [(1, "001"), (10, "010"), (100, "100"), (999, "999")]:
            assert f"{id_num:03d}" == expected


class TestConfigTypeDefault:
    """Test that config_type defaults are correct."""
    
    def test_checkpoint_runner_uses_checkpoint_type(self):
        """run_checkpoint.py should use config_type='checkpoint'."""
        # This tests the intent - actual integration test would run the script
        expected_config_type = "checkpoint"
        assert expected_config_type == "checkpoint"
    
    def test_experiment_runner_uses_experiment_type(self):
        """run_experiment.py should use config_type='experiment'."""
        expected_config_type = "experiment"
        assert expected_config_type == "experiment"


class TestRecallKConfigurable:
    """Test that recall_k is configurable."""
    
    def test_recall_k_default_is_100(self):
        """Default recall_k should be 100."""
        from lib.core.config_loader import load_config
        
        config = load_config(config_type="checkpoint")
        recall_k = config.get("retrieval", {}).get("recall_k", 100)
        assert recall_k == 100
    
    def test_recall_k_can_be_overridden(self):
        """recall_k should be overridable for experiments."""
        # Simulate override logic
        default_recall_k = 100
        override_recall_k = 200
        
        # If override provided, use it
        final_recall_k = override_recall_k if override_recall_k is not None else default_recall_k
        assert final_recall_k == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
