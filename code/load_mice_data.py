"""
Offline data loader for mice behavioral experiments.
Loads data from multiple subjects and saves to disk.
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import aind_dynamic_foraging_data_utils.code_ocean_utils as co
import aind_dynamic_foraging_multisession_analysis.multisession_load as ms_load
from aind_analysis_arch_result_access import han_pipeline


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / f"data_loading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_all_subject_ids() -> List[str]:
    """Get all available subject IDs from the session table."""
    logger = logging.getLogger(__name__)
    logger.info("Loading session table from han_pipeline...")
    
    df_han = han_pipeline.get_session_table(if_load_bpod=True)
    
    if 'subject_id' not in df_han.columns:
        raise KeyError(f"'subject_id' column not found. Available columns: {df_han.columns.tolist()}")
    
    all_subject_ids = df_han['subject_id'].unique().tolist()
    all_subject_ids = sorted(all_subject_ids)
    all_subject_ids = all_subject_ids[:200]
    logger.info(f"Found {len(all_subject_ids)} unique subjects")
    
    return all_subject_ids


def load_subject_data(
    subject_id: str,
    include_stages: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Load behavioral data for a single subject.
    
    Args:
        subject_id: Subject ID to load
        include_stages: List of stages to include (e.g., ["STAGE_FINAL", "GRADUATED"])
                       If None, all stages are included
    
    Returns:
        DataFrame with trials data, or None if no data found
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Querying docDB for subject {subject_id}")
    
    try:
        # Query for subject's sessions
        query_kwargs = {"modality": ["behavior"]}
        if include_stages:
            query_kwargs["stage"] = include_stages
        
        subject_results = co.get_subject_assets(subject_id, **query_kwargs)
        
        # Check if results were found
        if subject_results is None or len(subject_results) == 0:
            logger.warning(f"No data found for subject {subject_id}")
            return None
        
        # Sort by session name
        subject_results = subject_results.sort_values(
            by="session_name"
        ).reset_index(drop=True)
        
        num_sessions = len(subject_results)
        logger.info(f"Found {num_sessions} sessions for subject {subject_id}")
        
        # Add S3 locations
        logger.info(f"Getting S3 locations for subject {subject_id}")
        subject_results = co.add_s3_location(subject_results)
        
        # Load NWBs and create trials dataframe
        logger.info(f"Loading NWB files for subject {subject_id}")
        nwbs, df = ms_load.make_multisession_trials_df(subject_results["s3_nwb_location"])
        
        # Add subject identifier to the dataframe
        df["subject_id"] = subject_id
        
        num_trials = len(df)
        logger.info(f"Successfully loaded {num_trials} trials across {num_sessions} sessions for subject {subject_id}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for subject {subject_id}: {str(e)}", exc_info=True)
        return None


def load_multiple_subjects(
    subject_ids: List[str],
    include_stages: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load behavioral data for multiple subjects.
    
    Args:
        subject_ids: List of subject IDs to load
        include_stages: List of stages to include
    
    Returns:
        Combined DataFrame with all subjects' data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data for {len(subject_ids)} subjects")
    
    all_dfs = []
    successful_subjects = []
    failed_subjects = []
    subject_session_counts = {}
    
    for idx, subject_id in enumerate(subject_ids, start=1):
        logger.info("=" * 50)
        logger.info(f"Processing subject {subject_id} ({idx}/{len(subject_ids)})")
        df = load_subject_data(subject_id, include_stages=include_stages)
        
        if df is not None:
            all_dfs.append(df)
            successful_subjects.append(subject_id)
            # Count unique sessions for this subject
            if 'ses_idx' in df.columns:
                subject_session_counts[subject_id] = df['ses_idx'].nunique()
        else:
            failed_subjects.append(subject_id)
    
    # Combine all subjects into one dataframe
    if len(all_dfs) == 0:
        raise Exception(f"No data found for any subject IDs. Attempted: {subject_ids}")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    logger.info("=" * 50)
    logger.info("LOADING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Successfully loaded {len(successful_subjects)} subjects (out of {len(subject_ids)} attempted)")
    logger.info(f"Total trials: {len(combined_df)}")
    logger.info(f"Successful subjects: {successful_subjects}")
    
    # Log session counts per subject
    logger.info("\nSessions per subject:")
    for subject_id in successful_subjects:
        if subject_id in subject_session_counts:
            logger.info(f"  Subject {subject_id}: {subject_session_counts[subject_id]} sessions")
    
    if failed_subjects:
        logger.warning(f"\nFailed to load {len(failed_subjects)} subjects: {failed_subjects}")
    
    return combined_df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataframe to multiple formats."""
    logger = logging.getLogger(__name__)
    
    # Save as parquet (recommended for large datasets)
    parquet_path = output_path.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved data to {parquet_path}")
    
    # Save as CSV (for compatibility)
    # csv_path = output_path.with_suffix('.csv')
    # df.to_csv(csv_path, index=False)
    # logger.info(f"Saved data to {csv_path}")
    
    # Save as pickle (preserves dtypes perfectly)
    pkl_path = output_path.with_suffix('.pkl')
    df.to_pickle(pkl_path)
    logger.info(f"Saved data to {pkl_path}")
    
    # Print summary statistics
    logger.info("=" * 50)
    logger.info("DATAFRAME SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    if 'subject_id' in df.columns:
        logger.info(f"  Unique subjects: {df['subject_id'].nunique()}")
    if 'ses_idx' in df.columns:
        logger.info(f"  Total unique sessions: {df['ses_idx'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Load mice behavioral data offline and save to disk"
    )
    parser.add_argument(
        "--subject-ids",
        type=str,
        nargs="+",
        help="Specific subject IDs to load (space-separated). If not provided, uses first N subjects."
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=3,
        help="Number of subjects to load if --subject-ids not provided (default: 3)"
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Load all available subjects"
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=["STAGE_FINAL", "GRADUATED"],
        help="Filter by stages (e.g., --stages STAGE_FINAL GRADUATED)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/results",
        help="Output directory for saved data (default: /results)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename prefix (default: mice_behavioral_data_TIMESTAMP)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting data loading process")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Determine which subjects to load
        if args.subject_ids:
            subject_ids = args.subject_ids
            logger.info(f"Loading specified subjects: {subject_ids}")
        else:
            all_subject_ids = get_all_subject_ids()
            if args.all_subjects:
                subject_ids = all_subject_ids
                logger.info(f"Loading all {len(subject_ids)} subjects")
            else:
                subject_ids = all_subject_ids[:args.num_subjects]
                logger.info(f"Loading first {len(subject_ids)} subjects")
        
        # Load data
        combined_df = load_multiple_subjects(
            subject_ids=subject_ids,
            include_stages=args.stages
        )
        
        # Prepare output filename
        if args.output_name:
            output_name = args.output_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_name = f"mice_behavioral_data_{timestamp}"
        
        output_path = output_dir / output_name
        
        # Save data
        save_dataframe(combined_df, output_path)
        
        logger.info("Data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()