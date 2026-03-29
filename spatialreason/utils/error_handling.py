"""
Robust error handling and fallback mechanisms for the spatial reasoning agent pipeline.
Provides comprehensive error recovery, retry logic, and graceful failure handling.
"""

import time
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
from contextlib import contextmanager

from spatialreason.config.configuration_loader import ConfigurationLoader


class FailureReason(Enum):
    """Enumeration of failure reasons for categorized error handling."""
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    GPU_DEVICE_CONFLICT = "gpu_device_conflict"
    INSUFFICIENT_CLASSES = "insufficient_classes"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class ExecutionResult:
    """Standardized execution result with error handling metadata."""
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None
    failure_reason: Optional[FailureReason] = None
    retry_count: int = 0
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of planner-executor consistency validation."""
    is_valid: bool
    missing_classes: List[str]
    available_classes: List[str]
    required_classes: List[str]
    validation_message: str


class ClassificationToolLock:
    """Thread-safe lock mechanism for classification tool to prevent concurrent access."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._current_device = None
    
    @contextmanager
    def acquire_device(self, device_id: str):
        """Context manager for acquiring exclusive access to classification tool."""
        with self._lock:
            self._current_device = device_id
            try:
                yield device_id
            finally:
                self._current_device = None


class RobustErrorHandler:
    """
    Comprehensive error handling and fallback mechanisms for spatial reasoning pipeline.
    """
    
    def __init__(self, config_loader: Optional[ConfigurationLoader] = None):
        """
        Initialize the error handler with configuration.
        
        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader or ConfigurationLoader()
        self.logger = logging.getLogger(__name__)
        
        # Load error handling configuration
        self.error_config = self.config.get_config().get('defaults', {}).get('error_handling', {})
        
        # Configuration parameters
        self.max_retry_attempts = self.error_config.get('max_retry_attempts', 3)
        self.retry_backoff_factor = self.error_config.get('retry_backoff_factor', 2.0)
        self.retry_base_delay = self.error_config.get('retry_base_delay', 1.0)
        self.classification_timeout = self.error_config.get('classification_timeout_seconds', 30)
        self.tool_execution_timeout = self.error_config.get('tool_execution_timeout_seconds', 60)
        self.top_k_candidates = self.error_config.get('top_k_detection_candidates', 3)
        self.enable_graceful_failure = self.error_config.get('enable_graceful_failure', True)
        self.skip_on_insufficient_classes = self.error_config.get('skip_on_insufficient_classes', True)
        
        # Classification tool lock for preventing concurrent access
        self.classification_lock = ClassificationToolLock()
        
        # Execution statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retries_performed": 0,
            "failures_by_reason": {},
            "average_retry_count": 0.0
        }
        
        self.logger.info("🛡️ RobustErrorHandler initialized with comprehensive error handling")
    
    def validate_planner_executor_consistency(
        self, 
        planned_classes: List[str], 
        detection_results: Dict[str, Any],
        segmentation_results: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate consistency between planner expectations and executor results.
        
        Args:
            planned_classes: Classes expected by the planner
            detection_results: Results from detection tool execution
            segmentation_results: Optional results from segmentation tool
            
        Returns:
            ValidationResult with consistency analysis
        """
        try:
            # Extract detected classes from tool results
            detected_classes = set()
            
            # Process detection results
            if detection_results and detection_results.get("success", False):
                tool_result = detection_results.get("tool_result", {})
                if isinstance(tool_result, str):
                    tool_result = json.loads(tool_result)
                
                detections = tool_result.get("detections", [])
                for detection in detections:
                    if detection.get("confidence", 0) > 0.5:  # Filter by confidence
                        detected_classes.add(detection.get("class_name", "").lower())
            
            # Process segmentation results if available
            if segmentation_results and segmentation_results.get("success", False):
                seg_result = segmentation_results.get("tool_result", {})
                if isinstance(seg_result, str):
                    seg_result = json.loads(seg_result)
                
                segmentations = seg_result.get("segmentations", [])
                for seg in segmentations:
                    if seg.get("total_pixels", 0) > 100:  # Filter by area
                        detected_classes.add(seg.get("class_name", "").lower())
            
            # Normalize planned classes
            planned_classes_normalized = [cls.lower() for cls in planned_classes]
            
            # Find missing classes
            missing_classes = [cls for cls in planned_classes_normalized if cls not in detected_classes]
            available_classes = list(detected_classes)
            
            # Determine validation result
            is_valid = len(missing_classes) == 0
            
            if is_valid:
                message = f"✅ All planned classes found: {planned_classes_normalized}"
            else:
                message = f"⚠️ Missing classes: {missing_classes}. Available: {available_classes}"
            
            self.logger.info(f"🔍 Planner-Executor validation: {message}")
            
            return ValidationResult(
                is_valid=is_valid,
                missing_classes=missing_classes,
                available_classes=available_classes,
                required_classes=planned_classes_normalized,
                validation_message=message
            )
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                missing_classes=planned_classes,
                available_classes=[],
                required_classes=planned_classes,
                validation_message=f"Validation error: {str(e)}"
            )
    
    def execute_with_retry(
        self, 
        func, 
        *args, 
        max_attempts: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a function with retry logic and timeout handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            max_attempts: Maximum retry attempts (overrides default)
            timeout: Execution timeout in seconds
            **kwargs: Keyword arguments for the function
            
        Returns:
            ExecutionResult with execution outcome
        """
        max_attempts = max_attempts or self.max_retry_attempts
        start_time = time.time()
        
        for attempt in range(max_attempts):
            try:
                self.logger.info(f"🔄 Executing function (attempt {attempt + 1}/{max_attempts})")
                
                # Execute with timeout if specified
                if timeout:
                    result = self._execute_with_timeout(func, timeout, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                # Update statistics
                self.execution_stats["total_executions"] += 1
                self.execution_stats["successful_executions"] += 1
                if attempt > 0:
                    self.execution_stats["retries_performed"] += attempt
                
                self.logger.info(f"✅ Function executed successfully on attempt {attempt + 1}")
                
                return ExecutionResult(
                    success=True,
                    result=result,
                    retry_count=attempt,
                    execution_time=execution_time,
                    metadata={"attempts": attempt + 1}
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                failure_reason = self._classify_failure(e)
                
                self.logger.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                
                # Update failure statistics
                reason_key = failure_reason.value if failure_reason else "unknown"
                self.execution_stats["failures_by_reason"][reason_key] = \
                    self.execution_stats["failures_by_reason"].get(reason_key, 0) + 1
                
                # Check if we should retry
                if attempt < max_attempts - 1:
                    delay = self.retry_base_delay * (self.retry_backoff_factor ** attempt)
                    self.logger.info(f"🕐 Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    # Final failure
                    self.execution_stats["total_executions"] += 1
                    self.execution_stats["failed_executions"] += 1
                    
                    self.logger.error(f"❌ Function failed after {max_attempts} attempts")
                    
                    return ExecutionResult(
                        success=False,
                        error_message=str(e),
                        failure_reason=failure_reason,
                        retry_count=attempt + 1,
                        execution_time=execution_time,
                        metadata={"attempts": attempt + 1, "final_error": str(e)}
                    )
        
        # Should not reach here, but safety fallback
        return ExecutionResult(
            success=False,
            error_message="Unexpected execution path",
            failure_reason=FailureReason.TOOL_EXECUTION_FAILED,
            retry_count=max_attempts,
            execution_time=time.time() - start_time
        )

    def _execute_with_timeout(self, func, timeout: float, *args, **kwargs):
        """Execute function with timeout using threading."""
        import threading
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Function execution exceeded {timeout} seconds")

        if not exception_queue.empty():
            raise exception_queue.get()

        if not result_queue.empty():
            return result_queue.get()

        raise RuntimeError("Function execution completed but no result available")

    def _classify_failure(self, exception: Exception) -> Optional[FailureReason]:
        """Classify the type of failure based on exception details."""
        error_msg = str(exception).lower()

        if "cuda" in error_msg or "gpu" in error_msg or "device" in error_msg:
            return FailureReason.GPU_DEVICE_CONFLICT
        elif "timeout" in error_msg or isinstance(exception, TimeoutError):
            return FailureReason.TIMEOUT_EXCEEDED
        elif "insufficient" in error_msg or "missing" in error_msg:
            return FailureReason.INSUFFICIENT_CLASSES
        elif "semantic" in error_msg or "mismatch" in error_msg:
            return FailureReason.SEMANTIC_MISMATCH
        elif "resource" in error_msg or "memory" in error_msg:
            return FailureReason.RESOURCE_UNAVAILABLE
        elif "validation" in error_msg:
            return FailureReason.VALIDATION_FAILED
        else:
            return FailureReason.TOOL_EXECUTION_FAILED

    def handle_classification_tool_execution(
        self,
        classification_tool,
        device_id: str,
        *args,
        **kwargs
    ) -> ExecutionResult:
        """
        Handle classification tool execution with device conflict prevention.

        Args:
            classification_tool: Classification tool instance
            device_id: GPU device ID to use
            *args: Arguments for classification tool
            **kwargs: Keyword arguments for classification tool

        Returns:
            ExecutionResult with classification outcome
        """
        try:
            with self.classification_lock.acquire_device(device_id):
                self.logger.info(f"🔒 Acquired exclusive access to classification tool on device {device_id}")

                # Execute classification with timeout and retry
                result = self.execute_with_retry(
                    classification_tool.run,
                    *args,
                    timeout=self.classification_timeout,
                    **kwargs
                )

                self.logger.info(f"🔓 Released classification tool access")
                return result

        except Exception as e:
            self.logger.error(f"❌ Classification tool execution failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                failure_reason=self._classify_failure(e),
                metadata={"device_id": device_id}
            )

    def handle_detection_with_top_k_strategy(
        self,
        detection_tool,
        required_classes: List[str],
        *args,
        **kwargs
    ) -> ExecutionResult:
        """
        Handle detection with top-k strategy for better recall.

        Args:
            detection_tool: Detection tool instance
            required_classes: Classes required by the planner
            *args: Arguments for detection tool
            **kwargs: Keyword arguments for detection tool

        Returns:
            ExecutionResult with enhanced detection results
        """
        try:
            # Execute detection with retry
            detection_result = self.execute_with_retry(
                detection_tool.run,
                *args,
                **kwargs
            )

            if not detection_result.success:
                return detection_result

            # Process results to implement top-k strategy
            enhanced_result = self._apply_top_k_strategy(
                detection_result.result,
                required_classes
            )

            return ExecutionResult(
                success=True,
                result=enhanced_result,
                retry_count=detection_result.retry_count,
                execution_time=detection_result.execution_time,
                metadata={
                    "top_k_applied": True,
                    "required_classes": required_classes,
                    **detection_result.metadata
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Detection with top-k strategy failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                failure_reason=self._classify_failure(e)
            )

    def _apply_top_k_strategy(self, detection_result: Any, required_classes: List[str]) -> Any:
        """Apply top-k strategy to detection results for better recall."""
        try:
            if isinstance(detection_result, str):
                result_data = json.loads(detection_result)
            else:
                result_data = detection_result

            detections = result_data.get("detections", [])

            # Group detections by class and keep top-k for each
            class_detections = {}
            for detection in detections:
                class_name = detection.get("class_name", "").lower()
                if class_name not in class_detections:
                    class_detections[class_name] = []
                class_detections[class_name].append(detection)

            # Sort by confidence and keep top-k for each class
            enhanced_detections = []
            for class_name, class_dets in class_detections.items():
                # Sort by confidence descending
                sorted_dets = sorted(class_dets, key=lambda x: x.get("confidence", 0), reverse=True)
                # Keep top-k candidates
                top_k_dets = sorted_dets[:self.top_k_candidates]
                enhanced_detections.extend(top_k_dets)

            # Update result with enhanced detections
            result_data["detections"] = enhanced_detections
            result_data["top_k_strategy_applied"] = True
            result_data["top_k_value"] = self.top_k_candidates

            self.logger.info(f"📊 Applied top-{self.top_k_candidates} strategy: {len(enhanced_detections)} detections")

            return json.dumps(result_data) if isinstance(detection_result, str) else result_data

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply top-k strategy: {e}")
            return detection_result  # Return original result on failure

    def handle_graceful_failure(
        self,
        image_path: str,
        failure_reason: FailureReason,
        error_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle graceful failure with comprehensive logging and cleanup.

        Args:
            image_path: Path to the image being processed
            failure_reason: Categorized failure reason
            error_details: Additional error information

        Returns:
            Standardized failure response
        """
        if not self.enable_graceful_failure:
            raise RuntimeError(f"Graceful failure disabled. Error: {error_details}")

        failure_response = {
            "status": "failed_gracefully",
            "image_path": image_path,
            "failure_reason": failure_reason.value,
            "error_details": error_details,
            "timestamp": time.time(),
            "recovery_attempted": True
        }

        # Log failure with appropriate level
        if failure_reason == FailureReason.INSUFFICIENT_CLASSES and self.skip_on_insufficient_classes:
            self.logger.warning(f"⚠️ Skipping image due to insufficient classes: {Path(image_path).name}")
            failure_response["status"] = "skipped_insufficient_classes"
        else:
            self.logger.error(f"❌ Graceful failure for {Path(image_path).name}: {failure_reason.value}")

        # Update statistics
        reason_key = failure_reason.value
        self.execution_stats["failures_by_reason"][reason_key] = \
            self.execution_stats["failures_by_reason"].get(reason_key, 0) + 1

        return failure_response

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        total_executions = self.execution_stats["total_executions"]
        if total_executions > 0:
            success_rate = self.execution_stats["successful_executions"] / total_executions
            avg_retry_count = self.execution_stats["retries_performed"] / total_executions
        else:
            success_rate = 0.0
            avg_retry_count = 0.0

        return {
            "total_executions": total_executions,
            "successful_executions": self.execution_stats["successful_executions"],
            "failed_executions": self.execution_stats["failed_executions"],
            "success_rate": success_rate,
            "total_retries": self.execution_stats["retries_performed"],
            "average_retry_count": avg_retry_count,
            "failures_by_reason": self.execution_stats["failures_by_reason"],
            "configuration": {
                "max_retry_attempts": self.max_retry_attempts,
                "retry_backoff_factor": self.retry_backoff_factor,
                "classification_timeout": self.classification_timeout,
                "top_k_candidates": self.top_k_candidates,
                "graceful_failure_enabled": self.enable_graceful_failure
            }
        }
