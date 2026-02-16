"""
Shared CLI utilities for U-Time scripts.

This module provides common argument parsing functionality used across multiple
scripts to avoid code duplication.
"""


def add_wandb_arguments(parser):
    """
    Add Weights & Biases (wandb) arguments to an ArgumentParser.
    
    This function adds common wandb CLI arguments that are shared across
    train, evaluate, and predict scripts. These arguments can override
    the settings in hparams.yaml.
    
    Args:
        parser: ArgumentParser object to add arguments to
        
    Returns:
        ArgumentParser with wandb arguments added
    """
    # Global wandb arguments (work for all scripts)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (overrides hparams.yaml wandb.enabled setting).")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="W&B run ID to resume. If provided, resumes the existing run for logging. "
                             "If not provided and --wandb is set, creates a new run.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (overrides hparams.yaml)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team name (overrides hparams.yaml)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name (overrides hparams.yaml; auto-generated if not provided)")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group name for organizing related runs (overrides hparams.yaml)")
    parser.add_argument("--wandb-tags", nargs='*', type=str, default=None,
                        help="W&B tags for the run, space-separated (overrides hparams.yaml)")
    
    return parser


def merge_wandb_args_with_hparams(hparams, args):
    """
    Merge CLI wandb arguments with hparams configuration.
    
    CLI arguments take precedence over hparams.yaml settings. This function
    updates the wandb configuration in hparams with CLI overrides.
    
    Args:
        hparams: YAMLHParams object containing configuration
        args: Parsed command-line arguments (Namespace object)
        
    Returns:
        tuple: (wandb_config, cli_overrides) - merged config and dict of CLI overrides
    """
    # Get base wandb config from hparams
    wandb_config = hparams.get('wandb', {}).copy() if hparams.get('wandb') else {}
    
    # CLI flags override hparams settings
    if hasattr(args, 'wandb') and args.wandb:
        wandb_config['enabled'] = True
    
    # Override global settings with CLI arguments (if provided)
    cli_overrides = {}
    
    if hasattr(args, 'wandb_project') and args.wandb_project:
        cli_overrides['project'] = args.wandb_project
    
    if hasattr(args, 'wandb_entity') and args.wandb_entity:
        cli_overrides['entity'] = args.wandb_entity
    
    if hasattr(args, 'wandb_name') and args.wandb_name:
        cli_overrides['name'] = args.wandb_name
    
    if hasattr(args, 'wandb_group') and args.wandb_group:
        cli_overrides['group'] = args.wandb_group
    
    if hasattr(args, 'wandb_tags') and args.wandb_tags:
        cli_overrides['tags'] = args.wandb_tags
    
    # Apply CLI overrides to wandb config
    wandb_config.update(cli_overrides)
    
    return wandb_config, cli_overrides
