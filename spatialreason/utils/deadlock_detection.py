"""
Deadlock Detection and Recovery System for Spatial Reasoning Evaluation

This module provides robust timeout and loop detection mechanisms to prevent
the evaluation system from hanging on problematic files.
"""

import time
import signal
import threading
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
from collections import defaultdict, deque
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class DeadlockException(Exception):
    """Exception raised when deadlock is detected."""
    pass


class TimeoutException(Exception):
    """Exception raised when operation times out."""
    pass


class LoopDetectionException(Exception):
    """Exception raised when infinite loop is detected."""
    pass


class DeadlockDetector:
    """
    Comprehensive deadlock detection and recovery system.
    
    Features:
    - Configurable timeout limits
    - Loop detection through execution pattern monitoring
    - Graceful failure handling with standardized error outputs
    - Progress preservation for successful files
    """
    
    def __init__(self, 
                 timeout_seconds: int = 120,
                 loop_detection_window: int = 10,
                 max_identical_patterns: int = 3):
        """
        Initialize deadlock detector.
        
        Args:
            timeout_seconds: Maximum time allowed per file processing
            loop_detection_window: Number of recent operations to monitor for loops
            max_identical_patterns: Maximum identical patterns before declaring loop
        """
        self.timeout_seconds = timeout_seconds
        self.loop_detection_window = loop_detection_window
        self.max_identical_patterns = max_identical_patterns
        
        # Execution monitoring
        self.execution_history = deque(maxlen=loop_detection_window)
        self.error_patterns = defaultdict(int)
        self.start_time = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'timeout_failures': 0,
            'loop_failures': 0,
            'other_failures': 0
        }
    
    def timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise TimeoutException(f"Operation timed out after {self.timeout_seconds} seconds")
    
    def record_execution(self, operation: str, args: Dict = None, result: str = None):
        """Record an execution step for loop detection."""
        execution_record = {
            'timestamp': time.time(),
            'operation': operation,
            'args_hash': hash(str(sorted((args or {}).items()))),
            'result_hash': hash(result) if result else None,
            'error': result if result and 'error' in result.lower() else None
        }
        
        self.execution_history.append(execution_record)
        
        # Check for error patterns
        if execution_record['error']:
            error_key = f"{operation}:{execution_record['args_hash']}"
            self.error_patterns[error_key] += 1
            
            if self.error_patterns[error_key] >= self.max_identical_patterns:
                raise LoopDetectionException(
                    f"Detected infinite loop: {operation} with same args failed "
                    f"{self.error_patterns[error_key]} times consecutively"
                )
    
    def check_execution_loops(self):
        """Check for repetitive execution patterns indicating loops."""
        if len(self.execution_history) < self.max_identical_patterns:
            return
        
        # Check for identical operation sequences
        recent_ops = [record['operation'] for record in list(self.execution_history)[-self.max_identical_patterns:]]
        
        if len(set(recent_ops)) == 1:  # All operations are identical
            # Check if they have similar arguments
            recent_args = [record['args_hash'] for record in list(self.execution_history)[-self.max_identical_patterns:]]
            if len(set(recent_args)) <= 2:  # Very similar arguments
                raise LoopDetectionException(
                    f"Detected execution loop: {recent_ops[0]} repeated {len(recent_ops)} times "
                    f"with similar arguments"
                )
    
    def with_timeout_and_loop_detection(self, func: Callable) -> Callable:
        """
        Decorator to add timeout and loop detection to any function.
        
        Args:
            func: Function to wrap with deadlock detection
            
        Returns:
            Wrapped function with deadlock protection
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start_time = time.time()
            
            # Set up timeout signal
            old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            try:
                # Record function start
                self.record_execution(
                    operation=func.__name__,
                    args={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Record successful execution
                self.record_execution(
                    operation=f"{func.__name__}_success",
                    result=str(result)[:100] if result else None
                )
                
                self.stats['successful'] += 1
                return result
                
            except TimeoutException as e:
                self.stats['timeout_failures'] += 1
                logger.error(f"Timeout detected in {func.__name__}: {e}")
                raise DeadlockException(f"TIMEOUT: {e}")
                
            except LoopDetectionException as e:
                self.stats['loop_failures'] += 1
                logger.error(f"Loop detected in {func.__name__}: {e}")
                raise DeadlockException(f"LOOP: {e}")
                
            except Exception as e:
                self.stats['other_failures'] += 1
                # Record error execution
                self.record_execution(
                    operation=f"{func.__name__}_error",
                    result=str(e)[:100]
                )
                
                # Check if this might be part of a loop
                self.check_execution_loops()
                raise
                
            finally:
                # Reset alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                self.stats['total_processed'] += 1
        
        return wrapper
    
    def create_failure_prediction(self, 
                                  file_id: str, 
                                  error_type: str, 
                                  error_message: str,
                                  output_path: str) -> Dict[str, Any]:
        """
        Create standardized error prediction file for evaluation continuity.
        
        Args:
            file_id: Identifier of the failed file
            error_type: Type of failure (TIMEOUT, LOOP, ERROR)
            error_message: Detailed error message
            output_path: Path to save the error prediction
            
        Returns:
            Dictionary containing the error prediction structure
        """
        error_prediction = {
            "file_id": file_id,
            "status": f"FAILED_{error_type}",
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time(),
            "execution_time": time.time() - (self.start_time or time.time()),
            "prediction": f"ERROR: {error_type} - {error_message}",
            "confidence": 0.0,
            "metadata": {
                "deadlock_detected": True,
                "execution_history": list(self.execution_history)[-5:],  # Last 5 operations
                "stats": self.stats.copy()
            }
        }
        
        # Save error prediction to file
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(error_prediction, f, indent=2)
            
            logger.info(f"Created error prediction file: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create error prediction file: {e}")
        
        return error_prediction
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current detection statistics."""
        return {
            **self.stats,
            'success_rate': self.stats['successful'] / max(1, self.stats['total_processed']),
            'timeout_rate': self.stats['timeout_failures'] / max(1, self.stats['total_processed']),
            'loop_rate': self.stats['loop_failures'] / max(1, self.stats['total_processed'])
        }
    
    def reset(self):
        """Reset detector state for next file."""
        self.execution_history.clear()
        self.error_patterns.clear()
        self.start_time = None


# Global detector instance
_global_detector = DeadlockDetector()


def with_deadlock_protection(timeout_seconds: int = 120):
    """
    Decorator factory for adding deadlock protection to functions.
    
    Args:
        timeout_seconds: Timeout limit for the function
        
    Returns:
        Decorator function
    """
    def decorator(func):
        detector = DeadlockDetector(timeout_seconds=timeout_seconds)
        return detector.with_timeout_and_loop_detection(func)
    
    return decorator


def get_global_detector() -> DeadlockDetector:
    """Get the global deadlock detector instance."""
    return _global_detector


def reset_global_detector():
    """Reset the global deadlock detector."""
    _global_detector.reset()
