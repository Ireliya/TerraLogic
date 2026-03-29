"""
Robust Evaluation Wrapper with Deadlock Detection and Recovery

This module provides a robust wrapper around the evaluation pipeline that ensures
evaluation continuity even when individual files cause deadlocks or infinite loops.
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import traceback

from .deadlock_detection import DeadlockDetector, DeadlockException, get_global_detector

logger = logging.getLogger(__name__)


class RobustEvaluationManager:
    """
    Manager for robust evaluation with deadlock detection and recovery.
    
    Ensures that evaluation continues even when individual files fail,
    preserving partial results and maintaining evaluation continuity.
    """
    
    def __init__(self, 
                 output_dir: str,
                 timeout_per_file: int = 120,
                 max_retries: int = 1):
        """
        Initialize robust evaluation manager.
        
        Args:
            output_dir: Directory to save evaluation results
            timeout_per_file: Timeout limit per file in seconds
            max_retries: Maximum retry attempts for failed files
        """
        self.output_dir = Path(output_dir)
        self.timeout_per_file = timeout_per_file
        self.max_retries = max_retries
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir / "predictions"
        self.errors_dir = self.output_dir / "errors"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.predictions_dir, self.errors_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Evaluation statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed_timeout': 0,
            'failed_loop': 0,
            'failed_error': 0,
            'skipped_existing': 0,
            'start_time': time.time()
        }
        
        # Progress tracking
        self.processed_files = set()
        self.failed_files = {}
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for robust evaluation."""
        log_file = self.logs_dir / f"robust_evaluation_{int(time.time())}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def process_file_with_protection(self, 
                                     file_id: str,
                                     process_func: Callable,
                                     *args, **kwargs) -> Dict[str, Any]:
        """
        Process a single file with deadlock protection.
        
        Args:
            file_id: Unique identifier for the file
            process_func: Function to process the file
            *args, **kwargs: Arguments for the process function
            
        Returns:
            Dictionary containing processing results
        """
        # Check if already processed
        prediction_file = self.predictions_dir / f"{file_id}.json"
        error_file = self.errors_dir / f"{file_id}.json"
        
        if prediction_file.exists() or error_file.exists():
            logger.info(f"File {file_id} already processed, skipping")
            self.stats['skipped_existing'] += 1
            return {"status": "skipped", "file_id": file_id}
        
        # Initialize detector for this file
        detector = DeadlockDetector(timeout_seconds=self.timeout_per_file)
        
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                logger.info(f"Processing file {file_id} (attempt {retry_count + 1}/{self.max_retries + 1})")
                
                # Reset detector for each retry
                detector.reset()
                
                # Wrap the process function with deadlock protection
                protected_func = detector.with_timeout_and_loop_detection(process_func)
                
                # Execute the protected function with file_id as first argument
                result = protected_func(file_id, *args, **kwargs)
                
                # Save successful result
                success_result = {
                    "file_id": file_id,
                    "status": "success",
                    "result": result,
                    "timestamp": time.time(),
                    "retry_count": retry_count,
                    "execution_stats": detector.get_stats()
                }
                
                with open(prediction_file, 'w') as f:
                    json.dump(success_result, f, indent=2)
                
                logger.info(f"Successfully processed file {file_id}")
                self.stats['successful'] += 1
                self.processed_files.add(file_id)
                
                return success_result
                
            except DeadlockException as e:
                error_type = "TIMEOUT" if "TIMEOUT" in str(e) else "LOOP"
                logger.error(f"Deadlock detected for file {file_id}: {e}")
                
                # Create error prediction for evaluation continuity
                error_prediction = detector.create_failure_prediction(
                    file_id=file_id,
                    error_type=error_type,
                    error_message=str(e),
                    output_path=str(error_file)
                )
                
                # Update statistics
                if error_type == "TIMEOUT":
                    self.stats['failed_timeout'] += 1
                else:
                    self.stats['failed_loop'] += 1
                
                self.failed_files[file_id] = {
                    'error_type': error_type,
                    'error_message': str(e),
                    'retry_count': retry_count
                }
                
                # Don't retry deadlock failures - they're likely to repeat
                break
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                retry_count += 1
                if retry_count > self.max_retries:
                    # Create error prediction for final failure
                    error_prediction = detector.create_failure_prediction(
                        file_id=file_id,
                        error_type="ERROR",
                        error_message=str(e),
                        output_path=str(error_file)
                    )
                    
                    self.stats['failed_error'] += 1
                    self.failed_files[file_id] = {
                        'error_type': 'ERROR',
                        'error_message': str(e),
                        'retry_count': retry_count - 1
                    }
                    break
                else:
                    logger.info(f"Retrying file {file_id} (attempt {retry_count + 1})")
                    time.sleep(1)  # Brief pause before retry
        
        return {"status": "failed", "file_id": file_id}
    
    def process_file_list(self, 
                          file_list: List[str],
                          process_func: Callable,
                          get_args_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a list of files with robust error handling.
        
        Args:
            file_list: List of file identifiers to process
            process_func: Function to process each file
            get_args_func: Optional function to get arguments for each file
            
        Returns:
            Dictionary containing overall processing results
        """
        self.stats['total_files'] = len(file_list)
        logger.info(f"Starting robust evaluation of {len(file_list)} files")
        
        results = []
        
        for i, file_id in enumerate(file_list):
            logger.info(f"Processing file {i+1}/{len(file_list)}: {file_id}")
            
            try:
                # Get arguments for this file if function provided
                if get_args_func:
                    args, kwargs = get_args_func(file_id)
                else:
                    args, kwargs = (), {}

                # Process file with protection
                result = self.process_file_with_protection(
                    file_id=file_id,
                    process_func=process_func,
                    *args, **kwargs
                )
                
                results.append(result)
                
                # Log progress every 10 files
                if (i + 1) % 10 == 0:
                    self.log_progress()
                
            except Exception as e:
                logger.error(f"Critical error processing file {file_id}: {e}")
                # Continue with next file even if there's a critical error
                results.append({"status": "critical_error", "file_id": file_id, "error": str(e)})
                self.stats['failed_error'] += 1
        
        # Final statistics
        self.stats['end_time'] = time.time()
        self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
        
        self.log_final_summary()
        self.save_evaluation_report()
        
        return {
            'stats': self.stats,
            'results': results,
            'failed_files': self.failed_files
        }
    
    def log_progress(self):
        """Log current progress statistics."""
        total = self.stats['total_files']
        processed = self.stats['successful'] + self.stats['failed_timeout'] + \
                   self.stats['failed_loop'] + self.stats['failed_error']
        
        success_rate = self.stats['successful'] / max(1, processed) * 100
        
        logger.info(f"Progress: {processed}/{total} files processed "
                   f"({success_rate:.1f}% success rate)")
    
    def log_final_summary(self):
        """Log final evaluation summary."""
        stats = self.stats
        total = stats['total_files']
        
        logger.info("=" * 60)
        logger.info("ROBUST EVALUATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files: {total}")
        logger.info(f"Successful: {stats['successful']} ({stats['successful']/max(1,total)*100:.1f}%)")
        logger.info(f"Failed (timeout): {stats['failed_timeout']} ({stats['failed_timeout']/max(1,total)*100:.1f}%)")
        logger.info(f"Failed (loop): {stats['failed_loop']} ({stats['failed_loop']/max(1,total)*100:.1f}%)")
        logger.info(f"Failed (error): {stats['failed_error']} ({stats['failed_error']/max(1,total)*100:.1f}%)")
        logger.info(f"Skipped (existing): {stats['skipped_existing']} ({stats['skipped_existing']/max(1,total)*100:.1f}%)")
        logger.info(f"Total time: {stats['total_time']:.1f} seconds")
        logger.info(f"Average time per file: {stats['total_time']/max(1,total):.1f} seconds")
        logger.info("=" * 60)
    
    def save_evaluation_report(self):
        """Save detailed evaluation report."""
        report = {
            'evaluation_summary': self.stats,
            'failed_files': self.failed_files,
            'output_directories': {
                'predictions': str(self.predictions_dir),
                'errors': str(self.errors_dir),
                'logs': str(self.logs_dir)
            },
            'configuration': {
                'timeout_per_file': self.timeout_per_file,
                'max_retries': self.max_retries
            }
        }
        
        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {report_file}")


def create_robust_evaluation_wrapper(output_dir: str, 
                                     timeout_per_file: int = 120) -> RobustEvaluationManager:
    """
    Create a robust evaluation manager instance.
    
    Args:
        output_dir: Directory to save evaluation results
        timeout_per_file: Timeout limit per file in seconds
        
    Returns:
        RobustEvaluationManager instance
    """
    return RobustEvaluationManager(
        output_dir=output_dir,
        timeout_per_file=timeout_per_file
    )
