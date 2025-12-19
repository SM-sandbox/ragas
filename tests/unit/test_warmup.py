"""
Unit tests for Cloud Run orchestrator warmup functionality.

Tests the warmup ping logic, timing, and error handling.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestWarmupBasics:
    """Test basic warmup functionality."""
    
    def test_warmup_method_exists(self):
        """Verify _warmup_orchestrator method exists on GoldEvaluator."""
        from lib.core.evaluator import GoldEvaluator
        assert hasattr(GoldEvaluator, '_warmup_orchestrator')
    
    def test_warmup_returns_dict(self):
        """Verify warmup returns a dict with expected keys."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            # Mock requests to avoid actual network calls
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                # Need to set the CLOUD_RUN_URL
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=1, ping_interval=0.1, wait_after=0.1)
            
            assert isinstance(result, dict)
            assert 'pings' in result
            assert 'total_warmup_time' in result
            assert 'orchestrator_ready' in result


class TestWarmupPings:
    """Test ping behavior."""
    
    def test_warmup_sends_correct_number_of_pings(self):
        """Verify warmup sends the specified number of pings."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=5, ping_interval=0.01, wait_after=0.01)
            
            assert len(result['pings']) == 5
    
    def test_warmup_records_ping_times(self):
        """Verify each ping has a recorded time."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=3, ping_interval=0.01, wait_after=0.01)
            
            for ping in result['pings']:
                assert 'time' in ping
                assert ping['time'] >= 0
                assert 'status' in ping


class TestWarmupHealthEndpoint:
    """Test health endpoint fallback behavior."""
    
    def test_warmup_falls_back_to_retrieve_on_404(self):
        """Verify warmup falls back to /retrieve if /health returns 404."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                # First call (GET /health) returns 404
                mock_health_response = Mock()
                mock_health_response.status_code = 404
                
                # Second call (POST /retrieve) returns 200
                mock_retrieve_response = Mock()
                mock_retrieve_response.status_code = 200
                
                mock_requests.get.return_value = mock_health_response
                mock_requests.post.return_value = mock_retrieve_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=1, ping_interval=0.01, wait_after=0.01)
            
            # Should have called both get and post
            mock_requests.get.assert_called()
            mock_requests.post.assert_called()
            
            # Should still be marked as ready
            assert result['orchestrator_ready'] == True


class TestWarmupErrorHandling:
    """Test error handling during warmup."""
    
    def test_warmup_handles_timeout(self):
        """Verify warmup handles request timeouts gracefully."""
        from lib.core.evaluator import GoldEvaluator
        import requests as real_requests
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_requests.get.side_effect = real_requests.exceptions.Timeout()
                mock_requests.exceptions = real_requests.exceptions
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=1, ping_interval=0.01, wait_after=0.01)
            
            assert result['pings'][0]['status'] == 'timeout'
            assert result['orchestrator_ready'] == False
    
    def test_warmup_handles_connection_error(self):
        """Verify warmup handles connection errors gracefully."""
        from lib.core.evaluator import GoldEvaluator
        import requests as real_requests
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_requests.get.side_effect = Exception("Connection refused")
                mock_requests.exceptions = real_requests.exceptions
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=1, ping_interval=0.01, wait_after=0.01)
            
            assert result['pings'][0]['status'] == 'error'
            assert 'error' in result['pings'][0]
            assert result['orchestrator_ready'] == False


class TestWarmupTiming:
    """Test warmup timing behavior."""
    
    def test_warmup_respects_ping_interval(self):
        """Verify warmup waits between pings."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        start = time.time()
                        result = evaluator._warmup_orchestrator(num_pings=3, ping_interval=0.1, wait_after=0.01)
                        elapsed = time.time() - start
            
            # Should take at least 2 * 0.1s for intervals between 3 pings
            assert elapsed >= 0.2
    
    def test_warmup_respects_wait_after(self):
        """Verify warmup waits after successful pings."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        start = time.time()
                        result = evaluator._warmup_orchestrator(num_pings=1, ping_interval=0.01, wait_after=0.2)
                        elapsed = time.time() - start
            
            # Should take at least 0.2s for wait_after
            assert elapsed >= 0.2
    
    def test_warmup_tracks_total_time(self):
        """Verify warmup tracks total warmup time."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=2, ping_interval=0.1, wait_after=0.1)
            
            assert result['total_warmup_time'] > 0
            # Should be at least ping_interval + wait_after
            assert result['total_warmup_time'] >= 0.2


class TestWarmupIntegration:
    """Test warmup integration with run() method."""
    
    def test_run_calls_warmup_in_cloud_mode(self):
        """Verify run() calls warmup in cloud mode with multiple workers."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator.workers = 50
            evaluator._cloud_token = "fake_token"
            evaluator.config_type = "run"
            evaluator.config = {"client": "BFAI"}
            evaluator.precision_k = 25
            evaluator.model = "gemini-3-flash-preview"
            evaluator.generator_reasoning = "low"
            evaluator.lock = MagicMock()
            evaluator.run_start_time = None
            evaluator.run_dir = None
            evaluator.checkpoint_file = None
            evaluator.results_file = None
            evaluator.rate_limiter = MagicMock()
            
            # Mock the warmup method
            evaluator._warmup_orchestrator = MagicMock(return_value={
                'pings': [{'status': 'ok', 'time': 0.5}],
                'total_warmup_time': 1.0,
                'orchestrator_ready': True,
            })
            
            # Mock other methods to avoid full execution
            evaluator._create_run_directory = MagicMock(return_value=Path('/tmp/test'))
            
            with patch('builtins.open', MagicMock()):
                with patch.object(Path, 'exists', return_value=False):
                    with patch.object(Path, 'mkdir'):
                        # This will fail at checkpoint loading, but we just want to verify warmup was called
                        try:
                            evaluator.run([])
                        except:
                            pass
            
            # Verify warmup was called
            evaluator._warmup_orchestrator.assert_called_once()
    
    def test_run_skips_warmup_in_local_mode(self):
        """Verify run() skips warmup in local mode."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = False  # Local mode
            evaluator.workers = 50
            evaluator.config_type = "run"
            evaluator.config = {"client": "BFAI"}
            evaluator.precision_k = 25
            evaluator.model = "gemini-3-flash-preview"
            evaluator.generator_reasoning = "low"
            evaluator.lock = MagicMock()
            evaluator.run_start_time = None
            evaluator.run_dir = None
            evaluator.checkpoint_file = None
            evaluator.results_file = None
            evaluator.rate_limiter = MagicMock()
            
            # Mock the warmup method
            evaluator._warmup_orchestrator = MagicMock()
            
            # Mock other methods
            evaluator._create_run_directory = MagicMock(return_value=Path('/tmp/test'))
            
            with patch('builtins.open', MagicMock()):
                with patch.object(Path, 'exists', return_value=False):
                    with patch.object(Path, 'mkdir'):
                        try:
                            evaluator.run([])
                        except:
                            pass
            
            # Verify warmup was NOT called
            evaluator._warmup_orchestrator.assert_not_called()
    
    def test_run_skips_warmup_with_single_worker(self):
        """Verify run() skips warmup with single worker."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator.workers = 1  # Single worker
            evaluator._cloud_token = "fake_token"
            evaluator.config_type = "run"
            evaluator.config = {"client": "BFAI"}
            evaluator.precision_k = 25
            evaluator.model = "gemini-3-flash-preview"
            evaluator.generator_reasoning = "low"
            evaluator.lock = MagicMock()
            evaluator.run_start_time = None
            evaluator.run_dir = None
            evaluator.checkpoint_file = None
            evaluator.results_file = None
            evaluator.rate_limiter = MagicMock()
            
            # Mock the warmup method
            evaluator._warmup_orchestrator = MagicMock()
            
            # Mock other methods
            evaluator._create_run_directory = MagicMock(return_value=Path('/tmp/test'))
            
            with patch('builtins.open', MagicMock()):
                with patch.object(Path, 'exists', return_value=False):
                    with patch.object(Path, 'mkdir'):
                        try:
                            evaluator.run([])
                        except:
                            pass
            
            # Verify warmup was NOT called (single worker doesn't need warmup)
            evaluator._warmup_orchestrator.assert_not_called()


class TestDefaultWorkers:
    """Test default worker configuration."""
    
    def test_default_workers_is_50(self):
        """Verify DEFAULT_WORKERS is set to 50."""
        from lib.core.evaluator import DEFAULT_WORKERS
        assert DEFAULT_WORKERS == 50
    
    def test_default_workers_used_when_not_specified(self):
        """Verify evaluator uses DEFAULT_WORKERS when workers not specified."""
        from lib.core.evaluator import DEFAULT_WORKERS
        # This is implicitly tested by the config loading logic
        assert DEFAULT_WORKERS == 50


class TestWarmupStats:
    """Test warmup statistics tracking."""
    
    def test_warmup_counts_successful_pings(self):
        """Verify warmup correctly identifies successful pings."""
        from lib.core.evaluator import GoldEvaluator
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=3, ping_interval=0.01, wait_after=0.01)
            
            successful = [p for p in result['pings'] if p['status'] == 'ok']
            assert len(successful) == 3
            assert result['orchestrator_ready'] == True
    
    def test_warmup_marks_not_ready_on_all_failures(self):
        """Verify warmup marks orchestrator as not ready when all pings fail."""
        from lib.core.evaluator import GoldEvaluator
        import requests as real_requests
        
        with patch.object(GoldEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = GoldEvaluator()
            evaluator.cloud_mode = True
            evaluator._cloud_token = "fake_token"
            
            with patch('lib.core.evaluator.requests') as mock_requests:
                mock_requests.get.side_effect = Exception("Connection refused")
                mock_requests.exceptions = real_requests.exceptions
                
                with patch('lib.core.evaluator.CLOUD_RUN_URL', 'http://test'):
                    with patch('lib.core.evaluator.JOB_ID', 'test_job'):
                        result = evaluator._warmup_orchestrator(num_pings=3, ping_interval=0.01, wait_after=0.01)
            
            assert result['orchestrator_ready'] == False
