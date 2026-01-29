"""
Weights & Biases (wandb) integration utilities for U-Time/U-Sleep.

This module provides helper functions for initializing and managing wandb logging
throughout the U-Time training, evaluation, and prediction pipelines. It handles
graceful degradation when wandb is not installed or not enabled.
"""

import logging
import os
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

# Check if wandb is available
try:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbMetricsLogger = None
    WandbModelCheckpoint = None


def is_wandb_available() -> bool:
    """
    Check if wandb is installed and available for import.
    
    Returns:
        bool: True if wandb is available, False otherwise
    """
    return WANDB_AVAILABLE


def is_wandb_enabled(hparams, cli_wandb_flag: Optional[bool] = None) -> bool:
    """
    Determine if wandb should be enabled based on multiple sources:
    1. CLI flag (--wandb) - highest priority
    2. Environment variable WANDB_MODE
    3. YAML configuration (hparams.wandb.enabled)
    
    Args:
        hparams: YAMLHParams object containing configuration
        cli_wandb_flag: Optional CLI flag override (--wandb)
        
    Returns:
        bool: True if wandb should be enabled
    """
    # Check if wandb is available first
    if not is_wandb_available():
        return False
    
    # CLI flag has highest priority
    if cli_wandb_flag is not None:
        return cli_wandb_flag
    
    # Check environment variable
    wandb_mode = os.environ.get('WANDB_MODE', '').lower()
    if wandb_mode == 'disabled':
        return False
    
    # Check YAML configuration
    try:
        wandb_config = hparams.get('wandb', {})
        return wandb_config.get('enabled', False)
    except (KeyError, AttributeError):
        return False


def init_wandb_run(
    config: Dict[str, Any],
    hparams,
    datasets: Optional[List] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Initialize a wandb run with U-Time configuration.
    
    Merges YAML config with CLI overrides and environment variables.
    Logs hyperparameters and dataset metadata.
    
    Args:
        config: Wandb configuration dictionary from YAML
        hparams: Complete YAMLHParams object
        datasets: Optional list of dataset objects to log metadata
        cli_overrides: Optional dictionary with CLI argument overrides
                      (e.g., {'project': 'my-project', 'name': 'run-1'})
    
    Returns:
        wandb.run object if successful, None otherwise
    """
    if not is_wandb_available():
        logger.warning("wandb is not installed. Run 'pip install wandb' to enable experiment tracking.")
        return None
    
    try:
        # Merge configurations (CLI overrides take precedence)
        final_config = config.copy()
        if cli_overrides:
            final_config.update({k: v for k, v in cli_overrides.items() if v is not None})
        
        # Extract wandb.init parameters
        init_params = {
            'project': final_config.get('project', 'u-time'),
            'entity': final_config.get('entity'),
            'name': final_config.get('name'),
            'group': final_config.get('group'),
            'tags': final_config.get('tags', []),
            'notes': final_config.get('notes'),
            'config': _flatten_hparams(hparams),
            'save_code': final_config.get('log_code', True),
        }
        
        # Remove None values
        init_params = {k: v for k, v in init_params.items() if v is not None}
        
        # Initialize wandb run
        run = wandb.init(**init_params)
        
        if run:
            logger.info(f"Initialized wandb run: {run.name} (ID: {run.id})")
            logger.info(f"View run at: {run.url}")
            
            # Log dataset information if provided
            if datasets:
                log_dataset_info(datasets)
        
        return run
        
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        logger.info("Continuing without wandb logging...")
        return None


def _flatten_hparams(hparams, parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
    """
    Flatten nested hyperparameters dictionary for wandb config logging.
    
    Args:
        hparams: YAMLHParams or dictionary object
        parent_key: Parent key for nested items
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary with hierarchical keys
    """
    items = []
    
    # Convert YAMLHParams to dict if needed
    if hasattr(hparams, 'to_dict'):
        hparams_dict = hparams.to_dict()
    elif hasattr(hparams, '__dict__'):
        hparams_dict = hparams.__dict__
    else:
        hparams_dict = dict(hparams) if not isinstance(hparams, dict) else hparams
    
    for k, v in hparams_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # Skip private attributes and methods
        if k.startswith('_'):
            continue
            
        if isinstance(v, dict):
            items.extend(_flatten_hparams(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
            # For lists of dicts, create indexed entries
            for i, item in enumerate(v):
                items.extend(_flatten_hparams(item, f"{new_key}_{i}", sep=sep).items())
        else:
            # Convert to basic types that wandb can handle
            if v is not None:
                items.append((new_key, v))
    
    return dict(items)


def log_dataset_info(datasets: List) -> None:
    """
    Log dataset metadata to wandb as a table.
    
    Args:
        datasets: List of dataset objects (train/val/test)
    """
    if not is_wandb_available() or not wandb.run:
        return
    
    try:
        import pandas as pd
        
        dataset_info = []
        for dataset in datasets:
            info = {
                'dataset_id': getattr(dataset, 'identifier', 'unknown'),
                'n_samples': len(dataset) if hasattr(dataset, '__len__') else 'unknown',
                'n_subjects': len(getattr(dataset, 'pairs', [])) if hasattr(dataset, 'pairs') else 'unknown',
            }
            
            # Try to get channel info
            if hasattr(dataset, 'select_channels'):
                info['channels'] = ', '.join(dataset.select_channels)
            
            dataset_info.append(info)
        
        # Create wandb Table
        df = pd.DataFrame(dataset_info)
        table = wandb.Table(dataframe=df)
        wandb.log({"dataset_info": table})
        
        logger.info(f"Logged dataset information for {len(datasets)} dataset(s)")
        
    except Exception as e:
        logger.warning(f"Could not log dataset info: {e}")


def log_confusion_matrix(
    true: 'np.ndarray',
    pred: 'np.ndarray',
    class_names: List[str],
    epoch: int,
    dataset_id: Optional[str] = None
) -> None:
    """
    Create and log confusion matrix visualization to wandb.
    
    Args:
        true: Ground truth labels (1D array)
        pred: Predicted labels (1D array) or probabilities (2D array)
        class_names: List of class names for labeling
        epoch: Current epoch number
        dataset_id: Optional dataset identifier for multi-dataset training
    """
    if not is_wandb_available() or not wandb.run:
        return
    
    try:
        import numpy as np
        
        # Convert probabilities to labels if needed
        if len(pred.shape) > 1:
            pred = np.argmax(pred, axis=-1)
        
        # Flatten arrays
        true = true.flatten()
        pred = pred.flatten()
        
        # Create confusion matrix
        cm = wandb.plot.confusion_matrix(
            y_true=true,
            preds=pred,
            class_names=class_names
        )
        
        # Log with appropriate key
        key = f"confusion_matrix/{dataset_id}" if dataset_id else "confusion_matrix"
        wandb.log({key: cm, "epoch": epoch})
        
    except Exception as e:
        logger.warning(f"Could not log confusion matrix: {e}")


def log_validation_metrics(
    metrics_dict: Dict[str, float],
    dataset_ids: List[str],
    epoch: int
) -> None:
    """
    Log validation metrics from U-Time's Validation callback.
    
    Handles multi-dataset metrics with proper hierarchical naming.
    
    Args:
        metrics_dict: Dictionary of metrics {metric_name: value}
        dataset_ids: List of dataset identifiers
        epoch: Current epoch number
    """
    if not is_wandb_available() or not wandb.run:
        return
    
    try:
        # Log all metrics with epoch
        log_dict = metrics_dict.copy()
        log_dict['epoch'] = epoch
        wandb.log(log_dict)
        
    except Exception as e:
        logger.warning(f"Could not log validation metrics: {e}")


def create_wandb_callbacks(
    wandb_config: Dict[str, Any],
    hparams: Any,
    model_dir: str = "model"
) -> List:
    """
    Factory function to create all wandb callbacks based on configuration.
    
    Returns list of initialized wandb callbacks (official + custom).
    Handles graceful degradation if wandb is unavailable.
    
    Args:
        wandb_config: Wandb configuration dictionary from hparams
        hparams: Complete YAMLHParams object
        model_dir: Directory for saving model checkpoints
        
    Returns:
        List of initialized callback objects (empty list if wandb unavailable)
    """
    if not is_wandb_available():
        logger.info("wandb not available - skipping wandb callbacks")
        return []
    
    if not wandb.run:
        logger.warning("wandb run not initialized - cannot create callbacks")
        return []
    
    callbacks = []
    
    try:
        # 1. Add WandbMetricsLogger for standard metrics
        log_freq = wandb_config.get('log_freq', 'epoch')
        if isinstance(log_freq, int) and log_freq > 0:
            log_freq = 'epoch'  # For compatibility with Keras callback
        
        callbacks.append(WandbMetricsLogger(log_freq=log_freq))
        logger.info(f"Added WandbMetricsLogger (log_freq={log_freq})")
        
        # 2. Add WandbModelCheckpoint if model logging is enabled
        if wandb_config.get('log_model', False):
            checkpoint_path = os.path.join(model_dir, "wandb_checkpoint")
            callbacks.append(
                WandbModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_dice',
                    mode='max',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
            logger.info(f"Added WandbModelCheckpoint (path={checkpoint_path})")
        
        # 3. Add custom U-Time WandbCallback (will be implemented next)
        from utime.callbacks import WandbCallback
        callbacks.append(WandbCallback(config=wandb_config, hparams=hparams))
        logger.info("Added custom U-Time WandbCallback")
        
    except Exception as e:
        logger.error(f"Error creating wandb callbacks: {e}")
        return []
    
    return callbacks


def resume_wandb_run(run_id: str, project: Optional[str] = None) -> Optional[Any]:
    """
    Resume an existing wandb run for evaluation or prediction.
    
    Args:
        run_id: Wandb run ID to resume
        project: Optional project name
        
    Returns:
        wandb.run object if successful, None otherwise
    """
    if not is_wandb_available():
        logger.warning("wandb is not installed")
        return None
    
    try:
        run = wandb.init(
            id=run_id,
            project=project,
            resume="allow"
        )
        logger.info(f"Resumed wandb run: {run_id}")
        return run
        
    except Exception as e:
        logger.error(f"Failed to resume wandb run {run_id}: {e}")
        return None


def finish_wandb_run() -> None:
    """
    Properly finish the current wandb run.
    """
    if is_wandb_available() and wandb.run:
        try:
            wandb.finish()
            logger.info("Finished wandb run")
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {e}")
