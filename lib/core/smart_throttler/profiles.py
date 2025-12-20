"""
Step Profiles for Smart Throttler.

Provides token estimation based on step profiles with online calibration.
Maintains an EMA of observed tokens to adapt estimates over time.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class StepProfile:
    """
    Profile for a pipeline step with token estimation.
    
    Attributes:
        step_name: Name of the step (e.g., "bronze", "filtration")
        initial_estimated_prompt_tokens: Baseline token estimate
        variance_factor: Multiplier for conservative estimation (1.2 = 20% buffer)
        calibration_ema: Exponentially weighted moving average of observed tokens
        calibration_count: Number of observations used for calibration
    """
    step_name: str
    initial_estimated_prompt_tokens: int = 3000
    variance_factor: float = 1.2
    
    # Online calibration state
    calibration_ema: float = 0.0
    calibration_count: int = 0
    
    # EMA smoothing factor (0.1 = slow adaptation, 0.3 = faster)
    ema_alpha: float = 0.1
    
    def get_estimated_tokens(self, prompt_length: Optional[int] = None) -> int:
        """
        Get estimated token cost for a request.
        
        Uses calibrated EMA if available, otherwise falls back to baseline.
        
        Args:
            prompt_length: Optional actual prompt length in characters
            
        Returns:
            Estimated token count with variance buffer
        """
        if self.calibration_count >= 5 and self.calibration_ema > 0:
            # Use calibrated estimate
            base_estimate = self.calibration_ema
        elif prompt_length is not None:
            # Estimate from prompt length (rough: 4 chars per token)
            base_estimate = max(prompt_length // 4, self.initial_estimated_prompt_tokens)
        else:
            # Use baseline
            base_estimate = self.initial_estimated_prompt_tokens
        
        # Apply variance factor for conservative estimation
        return int(base_estimate * self.variance_factor)
    
    def record_observation(self, actual_tokens: int) -> None:
        """
        Record an observed token count for calibration.
        
        Updates the EMA with the new observation.
        
        Args:
            actual_tokens: Actual tokens used in the request
        """
        if actual_tokens <= 0:
            return
        
        if self.calibration_count == 0:
            self.calibration_ema = float(actual_tokens)
        else:
            # EMA update: new_ema = alpha * observation + (1 - alpha) * old_ema
            self.calibration_ema = (
                self.ema_alpha * actual_tokens + 
                (1 - self.ema_alpha) * self.calibration_ema
            )
        
        self.calibration_count += 1
    
    def get_calibration_multiplier(self) -> float:
        """
        Get the calibration multiplier (ratio of EMA to baseline).
        
        Returns:
            Multiplier (1.0 if not calibrated)
        """
        if self.calibration_count < 5 or self.calibration_ema <= 0:
            return 1.0
        return self.calibration_ema / self.initial_estimated_prompt_tokens


class ProfileManager:
    """
    Manages step profiles with thread-safe calibration.
    
    Provides token estimation and online calibration for all pipeline steps.
    """
    
    def __init__(self, profiles: Optional[Dict[str, StepProfile]] = None):
        """
        Initialize the profile manager.
        
        Args:
            profiles: Optional dict of step name to StepProfile
        """
        self._lock = threading.Lock()
        self._profiles: Dict[str, StepProfile] = profiles or {}
        
        # Default profile for unknown steps
        self._default_profile = StepProfile(
            step_name="_default",
            initial_estimated_prompt_tokens=3000,
            variance_factor=1.3,
        )
    
    def get_profile(self, step_name: str) -> StepProfile:
        """
        Get the profile for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            StepProfile for the step (or default if not found)
        """
        with self._lock:
            return self._profiles.get(step_name, self._default_profile)
    
    def register_profile(self, profile: StepProfile) -> None:
        """
        Register a step profile.
        
        Args:
            profile: StepProfile to register
        """
        with self._lock:
            self._profiles[profile.step_name] = profile
    
    def estimate_tokens(
        self, 
        step_name: str, 
        prompt_length: Optional[int] = None
    ) -> int:
        """
        Estimate tokens for a request.
        
        Args:
            step_name: Name of the step
            prompt_length: Optional prompt length in characters
            
        Returns:
            Estimated token count
        """
        profile = self.get_profile(step_name)
        return profile.get_estimated_tokens(prompt_length)
    
    def record_observation(self, step_name: str, actual_tokens: int) -> None:
        """
        Record an observed token count for calibration.
        
        Args:
            step_name: Name of the step
            actual_tokens: Actual tokens used
        """
        with self._lock:
            if step_name not in self._profiles:
                # Create a new profile for this step
                self._profiles[step_name] = StepProfile(
                    step_name=step_name,
                    initial_estimated_prompt_tokens=actual_tokens,
                )
            self._profiles[step_name].record_observation(actual_tokens)
    
    def get_calibration_stats(self) -> Dict[str, Dict]:
        """
        Get calibration statistics for all profiles.
        
        Returns:
            Dict mapping step name to calibration stats
        """
        with self._lock:
            stats = {}
            for name, profile in self._profiles.items():
                stats[name] = {
                    "baseline": profile.initial_estimated_prompt_tokens,
                    "ema": profile.calibration_ema,
                    "count": profile.calibration_count,
                    "multiplier": profile.get_calibration_multiplier(),
                    "current_estimate": profile.get_estimated_tokens(),
                }
            return stats
    
    def reset_calibration(self, step_name: Optional[str] = None) -> None:
        """
        Reset calibration for a step or all steps.
        
        Args:
            step_name: Step to reset, or None for all steps
        """
        with self._lock:
            if step_name:
                if step_name in self._profiles:
                    self._profiles[step_name].calibration_ema = 0.0
                    self._profiles[step_name].calibration_count = 0
            else:
                for profile in self._profiles.values():
                    profile.calibration_ema = 0.0
                    profile.calibration_count = 0


# Singleton instance
_profile_manager: Optional[ProfileManager] = None
_profile_manager_lock = threading.Lock()


def get_profile_manager() -> ProfileManager:
    """Get the singleton ProfileManager instance."""
    global _profile_manager
    with _profile_manager_lock:
        if _profile_manager is None:
            _profile_manager = ProfileManager()
        return _profile_manager


def reset_profile_manager() -> None:
    """Reset the singleton ProfileManager."""
    global _profile_manager
    with _profile_manager_lock:
        _profile_manager = None


def configure_profiles_from_config(config) -> ProfileManager:
    """
    Configure ProfileManager from ThrottlerConfig.
    
    Args:
        config: ThrottlerConfig with step_profiles
        
    Returns:
        Configured ProfileManager
    """
    global _profile_manager
    
    profiles = {}
    for step_name, profile_config in config.step_profiles.items():
        profiles[step_name] = StepProfile(
            step_name=step_name,
            initial_estimated_prompt_tokens=profile_config.initial_estimated_prompt_tokens,
            variance_factor=profile_config.variance_factor,
        )
    
    with _profile_manager_lock:
        _profile_manager = ProfileManager(profiles)
        return _profile_manager
