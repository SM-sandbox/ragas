#!/usr/bin/env python3
"""
Unit tests for gRPC/absl log suppression.

Ensures verbose INFO logs from Google libraries are suppressed
so evaluation progress is visible in terminal output.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLogSuppression:
    """Tests for gRPC/absl log suppression."""
    
    def test_grpc_verbosity_set_in_evaluator(self):
        """evaluator.py should set GRPC_VERBOSITY before imports."""
        # Read the evaluator file
        evaluator_path = Path(__file__).parent.parent.parent / "lib" / "core" / "evaluator.py"
        content = evaluator_path.read_text()
        
        # Check that GRPC_VERBOSITY is set BEFORE google imports
        grpc_line = content.find('GRPC_VERBOSITY')
        google_import = content.find('from google')
        
        assert grpc_line != -1, "GRPC_VERBOSITY should be set in evaluator.py"
        assert grpc_line < google_import, "GRPC_VERBOSITY must be set BEFORE google imports"
    
    def test_glog_minloglevel_set_in_evaluator(self):
        """evaluator.py should set GLOG_minloglevel before imports."""
        evaluator_path = Path(__file__).parent.parent.parent / "lib" / "core" / "evaluator.py"
        content = evaluator_path.read_text()
        
        glog_line = content.find('GLOG_minloglevel')
        google_import = content.find('from google')
        
        assert glog_line != -1, "GLOG_minloglevel should be set in evaluator.py"
        assert glog_line < google_import, "GLOG_minloglevel must be set BEFORE google imports"
    
    def test_checkpoint_runner_has_log_suppression(self):
        """run_checkpoint.py should suppress gRPC logs."""
        runner_path = Path(__file__).parent.parent.parent / "scripts" / "run_checkpoint.py"
        content = runner_path.read_text()
        
        assert 'GRPC_VERBOSITY' in content, "run_checkpoint.py should set GRPC_VERBOSITY"
        assert 'GLOG_minloglevel' in content, "run_checkpoint.py should set GLOG_minloglevel"
    
    def test_experiment_runner_has_log_suppression(self):
        """run_experiment.py should suppress gRPC logs."""
        runner_path = Path(__file__).parent.parent.parent / "scripts" / "run_experiment.py"
        content = runner_path.read_text()
        
        assert 'GRPC_VERBOSITY' in content, "run_experiment.py should set GRPC_VERBOSITY"
        assert 'GLOG_minloglevel' in content, "run_experiment.py should set GLOG_minloglevel"
    
    def test_env_vars_use_setdefault(self):
        """Log suppression should use setdefault to not override user settings."""
        evaluator_path = Path(__file__).parent.parent.parent / "lib" / "core" / "evaluator.py"
        content = evaluator_path.read_text()
        
        # Should use setdefault, not direct assignment
        assert 'os.environ.setdefault("GRPC_VERBOSITY"' in content, \
            "Should use setdefault for GRPC_VERBOSITY"
        assert 'os.environ.setdefault("GLOG_minloglevel"' in content, \
            "Should use setdefault for GLOG_minloglevel"


class TestLogSuppressionValues:
    """Tests for correct log suppression values."""
    
    def test_grpc_verbosity_is_error(self):
        """GRPC_VERBOSITY should be set to ERROR."""
        evaluator_path = Path(__file__).parent.parent.parent / "lib" / "core" / "evaluator.py"
        content = evaluator_path.read_text()
        
        assert '"GRPC_VERBOSITY", "ERROR"' in content, \
            "GRPC_VERBOSITY should be set to ERROR"
    
    def test_glog_minloglevel_is_2(self):
        """GLOG_minloglevel should be 2 (ERROR level)."""
        evaluator_path = Path(__file__).parent.parent.parent / "lib" / "core" / "evaluator.py"
        content = evaluator_path.read_text()
        
        assert '"GLOG_minloglevel", "2"' in content, \
            "GLOG_minloglevel should be 2 (ERROR level)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
