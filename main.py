import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re

# ============================================================================
# UNIVERSAL SCHEMA & CORE CONFIGURATION
# ============================================================================

UNIVERSAL_SCHEMA = {
    "event_id": ["incremental_id", "match_id", "event_id", "game_id", "fight id", "fixture_id", "contest_id"],
    "date": ["date", "match_date", "event_date", "fixture_date", "contest_date", "game_date"],
    "participant_name": ["fighter full name", "player", "participant", "fighter", "athlete", "name", "player_name"],
    "team1": ["hometeam", "team1", "home", "home_team", "home_side", "team_a"],
    "team2": ["awayteam", "team2", "away", "away_team", "away_side", "team_b"],
    "outcome": ["result", "winner", "ftr", "full_time_result", "match_result", "game_result"],
    
    # Team sports specific columns
    "score_home": ["fthg", "score_home", "home_score", "goals_home", "points_home", "runs_home", "home_goals"],
    "score_away": ["ftag", "score_away", "away_score", "goals_away", "points_away", "runs_away", "away_goals"],
    "shots_home": ["hs", "home_shots", "shots_home"],
    "shots_away": ["as", "away_shots", "shots_away"],
    "shots_target_home": ["hst", "home_shots_target", "shots_target_home"],
    "shots_target_away": ["ast", "away_shots_target", "shots_target_away"],
    "fouls_home": ["hf", "home_fouls", "fouls_home"],
    "fouls_away": ["af", "away_fouls", "fouls_away"],
    "corners_home": ["hc", "home_corners", "corners_home"],
    "corners_away": ["ac", "away_corners", "corners_away"],
    "yellows_home": ["hy", "home_yellows", "yellows_home"],
    "yellows_away": ["ay", "away_yellows", "yellows_away"],
    "reds_home": ["hr", "home_reds", "reds_home"],
    "reds_away": ["ar", "away_reds", "reds_away"],
    
    # Single odds (for combat sports)
    "odds": ["odds", "betting_line", "favorite_odds", "participant_odds"],
    
    # Multi-odds (for team sports)
    "odds_home": ["odds_home", "home_odds", "b365h", "psh", "odds_1", "1x2_1"],
    "odds_away": ["odds_away", "away_odds", "b365a", "psa", "odds_2", "1x2_2"],
    "odds_draw": ["odds_draw", "draw_odds", "b365d", "psd", "odds_x", "1x2_x"],
    
    # Other fields
    "winner_first": ["winner first name", "winner_first", "winnerfirstname", "winner_fname"],
    "winner_last": ["winner last name", "winner_last", "winnerlastname", "winner_lname"]
}

def auto_map_columns(raw_columns: List[str], schema: Dict = UNIVERSAL_SCHEMA) -> Dict[str, str]:
    """
    Intelligently map raw dataset columns to universal schema keys using fuzzy matching.
    
    Args:
        raw_columns: List of column names from the raw dataset
        schema: Universal schema mapping dictionary
    
    Returns:
        Dictionary mapping schema keys to actual column names
    """
    mapping = {}
    used_columns = set()
    
    # First pass: exact matches
    for key, synonyms in schema.items():
        for col in raw_columns:
            if col in used_columns:
                continue
            col_normalized = col.lower().replace("_", " ").replace("-", " ").strip()
            for synonym in synonyms:
                if col_normalized == synonym.lower():
                    mapping[key] = col
                    used_columns.add(col)
                    break
        if key in mapping:
            continue
    
    # Second pass: substring matches
    for key, synonyms in schema.items():
        if key in mapping:
            continue
        for col in raw_columns:
            if col in used_columns:
                continue
            col_normalized = col.lower().replace("_", " ").replace("-", " ")
            for synonym in synonyms:
                if synonym.lower() in col_normalized or col_normalized in synonym.lower():
                    mapping[key] = col
                    used_columns.add(col)
                    break
        if key in mapping:
            continue
    
    return mapping

# ============================================================================
# DATA STRUCTURE DETECTION & CLASSIFICATION
# ============================================================================

class DataStructureDetector:
    """Automatically detects the data processing engine based on column mappings."""
    
    @staticmethod
    def detect_processing_engine(column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Automatically determine which processing engine to use based on final column mappings.
        
        Returns:
            Dictionary containing engine type, structure details, and confidence
        """
        result = {
            "engine": None,
            "structure": None,
            "confidence": 0.0,
            "trigger_fields": [],
            "metadata": {}
        }
        
        # Check for Paired-Row Engine triggers
        paired_triggers = []
        if column_mapping.get('participant_name'):
            paired_triggers.append('participant_name')
        if column_mapping.get('winner_first') or column_mapping.get('winner_last'):
            paired_triggers.append('winner_info')
        
        # Check for Single-Row Engine triggers  
        single_triggers = []
        if column_mapping.get('team1') and column_mapping.get('team2'):
            single_triggers.append('teams')
        if column_mapping.get('outcome'):
            single_triggers.append('outcome')
        
        # Decision logic
        if paired_triggers and not single_triggers:
            result["engine"] = "paired_row"
            result["structure"] = "individual_participants"
            result["trigger_fields"] = paired_triggers
            result["confidence"] = 1.0 if len(paired_triggers) >= 2 else 0.7
            
        elif single_triggers and not paired_triggers:
            result["engine"] = "single_row" 
            result["structure"] = "team_based"
            result["trigger_fields"] = single_triggers
            result["confidence"] = 1.0 if 'teams' in single_triggers else 0.8
            
        elif paired_triggers and single_triggers:
            # Conflict - choose based on stronger signals
            if 'participant_name' in paired_triggers and len(paired_triggers) >= 2:
                result["engine"] = "paired_row"
                result["structure"] = "individual_participants"
                result["trigger_fields"] = paired_triggers
                result["confidence"] = 0.6  # Lower confidence due to conflict
            elif 'teams' in single_triggers:
                result["engine"] = "single_row"
                result["structure"] = "team_based" 
                result["trigger_fields"] = single_triggers
                result["confidence"] = 0.6
            else:
                result["engine"] = "unknown"
                result["structure"] = "ambiguous"
                result["confidence"] = 0.0
        else:
            result["engine"] = "unknown"
            result["structure"] = "insufficient_mapping"
            result["confidence"] = 0.0
        
        return result
    
    @staticmethod
    def validate_engine_requirements(column_mapping: Dict[str, str], engine_type: str) -> Tuple[bool, List[str]]:
        """Validate that required fields are mapped for the detected engine."""
        errors = []
        
        # Core requirements for both engines
        if not column_mapping.get('date'):
            errors.append("Date column is required for historical analysis")
        if not column_mapping.get('event_id'):
            errors.append("Event ID column is required to group participants/teams")
        
        # Engine-specific requirements
        if engine_type == "paired_row":
            if not column_mapping.get('participant_name'):
                errors.append("Participant Name is required for Paired-Row processing")
            if not (column_mapping.get('winner_first') or column_mapping.get('winner_last')):
                errors.append("Winner information (first or last name) is required for outcome determination")
                
        elif engine_type == "single_row":
            if not (column_mapping.get('team1') and column_mapping.get('team2')):
                errors.append("Both Team1 and Team2 are required for Single-Row processing")
            # Outcome is recommended but not strictly required
            
        return len(errors) == 0, errors

# ============================================================================
# ENHANCED COLUMN MAPPING & VALIDATION
# ============================================================================

class EnhancedColumnMapper:
    """Advanced column mapping with intelligent suggestions and validation."""
    
    @staticmethod
    def get_suggestions(df: pd.DataFrame, target_field: str) -> List[str]:
        """Get intelligent column suggestions for a target field."""
        if target_field not in UNIVERSAL_SCHEMA:
            return []
        
        synonyms = UNIVERSAL_SCHEMA[target_field]
        suggestions = []
        
        for col in df.columns:
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            for synonym in synonyms:
                if synonym.lower() in col_lower:
                    suggestions.append(col)
                    break
        
        return suggestions
    
    @staticmethod
    def validate_mapping(df: pd.DataFrame, column_mapping: Dict[str, str], 
                        structure_type: str) -> Tuple[bool, List[str]]:
        """Validate column mapping completeness and correctness."""
        errors = []
        
        # Required fields by structure type
        if structure_type == "paired_participants":
            required = ['participant_name', 'event_id', 'date']
            recommended = ['winner_first', 'winner_last']
        elif structure_type == "single_match":
            required = ['team1', 'team2', 'date']
            recommended = ['outcome', 'event_id']
        else:
            required = ['date']
            recommended = []
        
        # Check required fields
        for field in required:
            if column_mapping.get(field, 'None') == 'None':
                errors.append(f"Required field '{field}' is not mapped")
        
        # Check if mapped columns exist
        for field, col_name in column_mapping.items():
            if col_name != 'None' and col_name not in df.columns:
                errors.append(f"Mapped column '{col_name}' for field '{field}' does not exist")
        
        return len(errors) == 0, errors

# ============================================================================
# UNIVERSAL SPORTS PROCESSOR ARCHITECTURE
# ============================================================================

class BaseFeatureGenerator(ABC):
    """Abstract base class for sport-specific feature generation."""
    
    @abstractmethod
    def generate_historical_features(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                   config: Dict[str, Any]) -> pd.DataFrame:
        """Generate historical features for the specific sport type."""
        pass
    
    @abstractmethod
    def calculate_participant_stats(self, participant_history: pd.DataFrame, 
                                  config: Dict[str, Any]) -> pd.Series:
        """Calculate statistical features for a single participant's history."""
        pass

class IndividualCombatProcessor(BaseFeatureGenerator):
    """Processor for individual combat sports (MMA, Boxing, etc.)."""
    
    def generate_historical_features(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                   config: Dict[str, Any]) -> pd.DataFrame:
        """Generate features for paired-participant combat sports."""
        processed_events = []
        total_rows = len(df)
        
        participant_col = column_mapping['participant_name']
        event_id_col = column_mapping['event_id']
        date_col = column_mapping['date']
        odds_col = column_mapping.get('odds')
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (i, participant_row) in enumerate(df.iterrows()):
            status_text.text(f"Processing event {idx+1}/{total_rows}")
            
            event_id = participant_row[event_id_col]
            event_date = participant_row[date_col]
            participant_name = participant_row[participant_col]
            
            # Find opponent in the same event
            opponent_query = df[(df[event_id_col] == event_id) & (df.index != i)]
            if opponent_query.empty:
                continue
            
            opponent_row = opponent_query.iloc[0]
            opponent_name = opponent_row[participant_col]
            
            # Get historical data for both participants (before current event)
            participant_history = df[(df[participant_col] == participant_name) & 
                                   (df[date_col] < event_date)]
            opponent_history = df[(df[participant_col] == opponent_name) & 
                                (df[date_col] < event_date)]
            
            # Generate stats for both participants
            participant_stats = self.calculate_participant_stats(participant_history, config)
            opponent_stats = self.calculate_participant_stats(opponent_history, config)
            
            # Calculate difference features
            diff_stats = (participant_stats - opponent_stats).add_prefix('diff_')
            
            # Process odds if available
            odds_features = self._process_odds(participant_row, opponent_row, odds_col)
            
            # Combine all features
            event_features = self._combine_event_features(
                participant_row, opponent_row, participant_stats, 
                opponent_stats, diff_stats, odds_features, column_mapping
            )
            
            processed_events.append(event_features)
            progress_bar.progress((idx + 1) / total_rows)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(processed_events).fillna(0)
    
    def calculate_participant_stats(self, participant_history: pd.DataFrame, 
                                  config: Dict[str, Any]) -> pd.Series:
        """Calculate historical statistics for a combat sports participant."""
        all_stats = {}
        
        if participant_history.empty:
            return pd.Series(all_stats)
        
        aggregation_types = config.get('aggregation_types', ['Average'])
        lookback_windows = config.get('lookback_windows', [0, 5, 10])
        outcome_types = config.get('outcome_types', ['Wins', 'Losses'])
        betting_statuses = config.get('betting_statuses', ['Overall'])
        stat_columns = config.get('stat_columns', [])
        odds_col = config.get('odds_column')
        
        for window in lookback_windows:
            window_data = participant_history.tail(window) if window > 0 else participant_history
            
            # Basic fight metrics
            total_fights = len(window_data)
            wins = window_data[window_data['Winner']] if 'Winner' in window_data.columns else pd.DataFrame()
            losses = window_data[~window_data['Winner']] if 'Winner' in window_data.columns else pd.DataFrame()
            
            all_stats[f'total_fights_{window}'] = total_fights
            all_stats[f'wins_{window}'] = len(wins)
            all_stats[f'losses_{window}'] = len(losses)
            all_stats[f'win_rate_{window}'] = len(wins) / total_fights if total_fights > 0 else 0
            
            # Generate stats by outcome and betting status
            self._generate_segmented_stats(window_data, wins, losses, all_stats, window, 
                                         aggregation_types, outcome_types, betting_statuses, 
                                         stat_columns, odds_col)
        
        return pd.Series(all_stats).fillna(0)
    
    def _process_odds(self, participant_row: pd.Series, opponent_row: pd.Series, 
                     odds_col: Optional[str]) -> Dict[str, float]:
        """Process betting odds for both participants using the specified formula."""
        odds_features = {}
        
        if odds_col and odds_col in participant_row.index:
            raw_p_odds = participant_row.get(odds_col, 0)
            raw_o_odds = opponent_row.get(odds_col, 0)
            
            # Apply the specified odds conversion formula
            p_odds = self._convert_odds(raw_p_odds)
            o_odds = self._convert_odds(raw_o_odds)
            
            # Calculate probability and difference
            p_prob = self._odds_to_probability(raw_p_odds)
            o_prob = self._odds_to_probability(raw_o_odds)
            
            odds_features.update({
                'participant_odds': p_odds,
                'opponent_odds': o_odds,
                'participant_prob': p_prob,
                'opponent_prob': o_prob,
                'odds_difference': p_odds - o_odds,
                'raw_participant_odds': raw_p_odds,
                'raw_opponent_odds': raw_o_odds
            })
        
        return odds_features
    
    def _convert_odds(self, raw_odds: float) -> float:
        """
        Convert raw odds using the specified formula:
        - If positive (or zero): keep exactly as is
        - If negative: round(10000 / abs(raw_odds))
        
        Examples:
        - Raw odds of +236 becomes 236
        - Raw odds of -345 becomes round(10000 / 345) = 29
        """
        if raw_odds >= 0:
            return raw_odds
        else:
            return round(10000 / abs(raw_odds))
    
    def _odds_to_probability(self, odds: float) -> float:
        """Convert betting odds to implied probability."""
        if odds == 0:
            return 0.5
        elif odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def _generate_segmented_stats(self, window_data: pd.DataFrame, wins: pd.DataFrame, 
                                losses: pd.DataFrame, all_stats: Dict, window: int,
                                aggregation_types: List[str], outcome_types: List[str],
                                betting_statuses: List[str], stat_columns: List[str],
                                odds_col: Optional[str]) -> None:
        """Generate statistics segmented by outcome and betting status."""
        
        segments = {}
        if 'Overall' in betting_statuses:
            if 'Wins' in outcome_types:
                segments['win'] = wins
            if 'Losses' in outcome_types:
                segments['loss'] = losses
        
        # Add betting status segments if odds available
        if odds_col and odds_col in window_data.columns:
            if 'As Favorite' in betting_statuses:
                favorite_data = window_data[window_data[odds_col] < 0]
                if 'Wins' in outcome_types:
                    segments['win_favorite'] = favorite_data[favorite_data.get('Winner', False)]
                if 'Losses' in outcome_types:
                    segments['loss_favorite'] = favorite_data[~favorite_data.get('Winner', True)]
            
            if 'As Underdog' in betting_statuses:
                underdog_data = window_data[window_data[odds_col] >= 0]
                if 'Wins' in outcome_types:
                    segments['win_underdog'] = underdog_data[underdog_data.get('Winner', False)]
                if 'Losses' in outcome_types:
                    segments['loss_underdog'] = underdog_data[~underdog_data.get('Winner', True)]
        
        # Calculate aggregated statistics for each segment
        agg_map = {'Average': 'mean', 'Sum': 'sum', 'Median': 'median'}
        for agg_type in aggregation_types:
            agg_func = agg_map[agg_type]
            agg_name = 'avg' if agg_type == 'Average' else agg_type.lower()
            
            for col in stat_columns:
                if col in window_data.columns:
                    for segment_name, segment_data in segments.items():
                        stat_name = f'{segment_name}_{agg_name}_{col.replace(" ", "_")}_{window}'
                        all_stats[stat_name] = segment_data[col].agg(agg_func) if not segment_data.empty else 0
    
    def _combine_event_features(self, participant_row: pd.Series, opponent_row: pd.Series,
                              participant_stats: pd.Series, opponent_stats: pd.Series,
                              diff_stats: pd.Series, odds_features: Dict[str, float],
                              column_mapping: Dict[str, str]) -> pd.Series:
        """Combine all features into a single event record."""
        
        # Basic event info
        event_info = pd.Series({
            'event_id': participant_row[column_mapping['event_id']],
            'date': participant_row[column_mapping['date']],
            'participant_name': participant_row[column_mapping['participant_name']],
            'opponent_name': opponent_row[column_mapping['participant_name']],
            'participant_won': int(participant_row.get('Winner', False))
        })
        
        # Add suffixes to distinguish participant vs opponent stats
        participant_stats = participant_stats.add_suffix('_participant')
        opponent_stats = opponent_stats.add_suffix('_opponent')
        
        # Combine all features
        return pd.concat([
            event_info,
            participant_stats,
            opponent_stats,
            diff_stats,
            pd.Series(odds_features)
        ])

class TeamSportsProcessor(BaseFeatureGenerator):
    """Processor for team-based sports (Football, Cricket, Basketball, etc.)."""
    
    def generate_historical_features(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                   config: Dict[str, Any]) -> pd.DataFrame:
        """Generate comprehensive historical features matching the expected output format."""
        # Required columns
        date_col = column_mapping.get('date')
        home_col = column_mapping.get('team1')
        away_col = column_mapping.get('team2') 
        outcome_col = column_mapping.get('outcome')
        
        if not all([date_col, home_col, away_col]):
            st.error("❌ Missing required columns for team sports processing")
            return df
        
        # Convert date to datetime and sort
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
        
        # Get statistical columns mapping
        stat_columns = self._get_statistical_columns(column_mapping)
        
        # Get lookback windows
        lookback_windows = config.get('lookback_windows', [5, 15, 38])
        
        # Initialize result list
        enhanced_matches = []
        total_rows = len(df)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each match
        for i in range(len(df)):
            match_row = df.iloc[i]
            status_text.text(f"Processing match {i+1}/{total_rows}")
            
            if date_col and home_col and away_col:
                match_date = match_row[date_col]
                home_team = match_row[home_col]
                away_team = match_row[away_col]
                
                # Calculate historical stats for both teams
                home_stats = self._calculate_team_historical_stats(
                    df, home_team, match_date, i, home_col, away_col, 
                    outcome_col or '', stat_columns, lookback_windows
                )
                
                away_stats = self._calculate_team_historical_stats(
                    df, away_team, match_date, i, home_col, away_col, 
                    outcome_col or '', stat_columns, lookback_windows
                )
                
                # Create enhanced match record
                enhanced_match = match_row.to_dict()
                
                # Add home team stats with prefix
                for key, value in home_stats.items():
                    enhanced_match[f"Home_{key}"] = value
                    
                # Add away team stats with prefix 
                for key, value in away_stats.items():
                    enhanced_match[f"Away_{key}"] = value
                
                enhanced_matches.append(enhanced_match)
                
            progress_bar.progress((i + 1) / total_rows)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(enhanced_matches)
    
    def _get_statistical_columns(self, column_mapping: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """Get available statistical columns for processing."""
        stat_cols = {}
        
        # Core stats mapping: stat_name -> (home_col, away_col)
        stat_mapping = {
            'GoalsFor': ('score_home', 'score_away'),
            'Shots': ('shots_home', 'shots_away'),
            'ShotsOnTarget': ('shots_target_home', 'shots_target_away'),
            'Fouls': ('fouls_home', 'fouls_away'),
            'Corners': ('corners_home', 'corners_away'),
            'Yellows': ('yellows_home', 'yellows_away'),
            'Reds': ('reds_home', 'reds_away')
        }
        
        for stat_name, (home_key, away_key) in stat_mapping.items():
            home_col = column_mapping.get(home_key)
            away_col = column_mapping.get(away_key)
            if home_col and away_col:
                stat_cols[stat_name] = (home_col, away_col)
        
        return stat_cols
    
    def _calculate_team_historical_stats(self, df: pd.DataFrame, team_name: str, 
                                       match_date: pd.Timestamp, current_idx: int,
                                       home_col: str, away_col: str, outcome_col: str,
                                       stat_columns: Dict[str, Tuple[str, str]], 
                                       lookback_windows: List[int]) -> Dict[str, Any]:
        """Calculate comprehensive historical statistics for a team."""
        # Get team's historical matches (before current match)
        historical_matches = df[
            ((df[home_col] == team_name) | (df[away_col] == team_name))
        ].iloc[:current_idx]  # Only matches before current one
        
        if historical_matches.empty:
            return self._get_empty_stats(stat_columns, lookback_windows)
        
        # Process each historical match to extract team-specific stats
        team_history = []
        for _, match in historical_matches.iterrows():
            is_home = match[home_col] == team_name
            
            # Extract team's stats for this match
            match_stats = {
                'Date': match.get('Date', match_date),
                'Team': team_name
            }
            
            # Determine result for this team
            team_result = self._get_team_result(match, team_name, home_col, away_col, outcome_col)
            match_stats['FTR'] = team_result
            
            # Add statistical data
            for stat_name, (home_stat_col, away_stat_col) in stat_columns.items():
                if is_home:
                    team_value = match.get(home_stat_col, 0)
                    opponent_value = match.get(away_stat_col, 0)
                else:
                    team_value = match.get(away_stat_col, 0) 
                    opponent_value = match.get(home_stat_col, 0)
                
                match_stats[stat_name] = float(team_value) if pd.notna(team_value) else 0.0
                match_stats[f'{stat_name}Against'] = float(opponent_value) if pd.notna(opponent_value) else 0.0
            
            # Add win/draw/loss indicators
            match_stats['Win'] = 1 if team_result == 'W' else 0
            match_stats['Draw'] = 1 if team_result == 'D' else 0
            match_stats['Loss'] = 1 if team_result == 'L' else 0
            
            team_history.append(match_stats)
        
        if not team_history:
            return self._get_empty_stats(stat_columns, lookback_windows)
        
        # Convert to DataFrame for easier processing
        team_df = pd.DataFrame(team_history)
        
        # Calculate statistics for different lookback windows
        final_stats = {}
        
        # Add current match info for reference
        final_stats['Date'] = match_date
        final_stats['Team'] = team_name
        
        # Overall stats (all history)
        self._add_window_stats(final_stats, team_df, '', stat_columns)
        
        # Windowed stats
        for window in lookback_windows:
            if window > 0:
                window_df = team_df.tail(window) if len(team_df) >= window else team_df
                self._add_window_stats(final_stats, window_df, f'_L{window}', stat_columns)
        
        return final_stats
    
    def _add_window_stats(self, stats_dict: Dict[str, Any], window_df: pd.DataFrame, 
                         suffix: str, stat_columns: Dict[str, Tuple[str, str]]) -> None:
        """Add statistical calculations for a specific window."""
        if window_df.empty:
            return
        
        # Basic outcome stats
        stats_dict[f'Win{suffix}'] = int(window_df['Win'].sum())
        stats_dict[f'Draw{suffix}'] = int(window_df['Draw'].sum())
        stats_dict[f'Loss{suffix}'] = int(window_df['Loss'].sum())
        
        # Statistical measures for each column
        for stat_name in stat_columns.keys():
            if stat_name in window_df.columns:
                stats_dict[f'{stat_name}{suffix}'] = float(window_df[stat_name].sum())
                
            # Goals/stats against calculation
            against_col = f'{stat_name}Against'
            if against_col in window_df.columns:
                stats_dict[f'{stat_name}Against{suffix}'] = float(window_df[against_col].sum())
    
    def _get_team_result(self, match_row: pd.Series, team_name: str, 
                        home_col: str, away_col: str, outcome_col: str) -> str:
        """Determine team result (W/D/L) from match outcome."""
        if not outcome_col or outcome_col not in match_row:
            return 'D'
            
        is_home = match_row[home_col] == team_name
        outcome = str(match_row[outcome_col]).upper()
        
        if outcome == 'D':
            return 'D'
        elif (outcome == 'H' and is_home) or (outcome == 'A' and not is_home):
            return 'W'
        else:
            return 'L'
    
    def _get_empty_stats(self, stat_columns: Dict[str, Tuple[str, str]], 
                        lookback_windows: List[int]) -> Dict[str, Any]:
        """Return empty stats structure when no historical data available."""
        empty_stats: Dict[str, Any] = {
            'Date': '',
            'Team': '',
            'FTR': 'D'
        }
        
        # Basic win/draw/loss
        for outcome in ['Win', 'Draw', 'Loss']:
            empty_stats[outcome] = 0
            for window in lookback_windows:
                if window > 0:
                    empty_stats[f'{outcome}_L{window}'] = 0
        
        # Statistical columns
        for stat_name in stat_columns.keys():
            empty_stats[stat_name] = 0.0
            empty_stats[f'{stat_name}Against'] = 0.0
            for window in lookback_windows:
                if window > 0:
                    empty_stats[f'{stat_name}_L{window}'] = 0.0
                    empty_stats[f'{stat_name}Against_L{window}'] = 0.0
        
        return empty_stats
    
    def calculate_participant_stats(self, participant_history: pd.DataFrame, 
                                  config: Dict[str, Any]) -> pd.Series:
        """Calculate participant statistics (compatibility method)."""
        if participant_history.empty:
            return pd.Series({'matches': 0, 'wins': 0, 'avg_score': 0.0})
        
        # Basic aggregations
        stats = {
            'matches_played': len(participant_history),
            'wins': len(participant_history[participant_history.get('result', False)]),
            'avg_score': participant_history.get('score', pd.Series([0])).mean()
        }
        
        return pd.Series(stats)

class RacingSportsProcessor(BaseFeatureGenerator):
    """Processor for racing sports (F1, Horse Racing, etc.)."""
    
    def generate_historical_features(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                   config: Dict[str, Any]) -> pd.DataFrame:
        """Generate features for racing sports (multi-participant events)."""
        # This would be implemented for racing sports where multiple participants
        # compete in the same event (like F1 races, horse races, etc.)
        # For now, return empty DataFrame as placeholder
        st.warning("Racing sports processor not yet implemented. Please use individual or team sports structure.")
        return pd.DataFrame()
    
    def calculate_participant_stats(self, participant_history: pd.DataFrame, 
                                  config: Dict[str, Any]) -> pd.Series:
        """Calculate statistical features for racing participants."""
        return pd.Series()

class UniversalSportsProcessor:
    """Main processor with automatic dual-engine detection and routing."""
    
    def __init__(self):
        self.paired_row_processor = IndividualCombatProcessor()
        self.single_row_processor = TeamSportsProcessor()
    
    def process_data(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                    detection_result: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        """Process data using the automatically detected engine."""
        
        engine_type = detection_result.get("engine", "unknown")
        
        if engine_type == "unknown":
            raise ValueError("Cannot determine processing engine. Please check column mappings.")
        
        # Preprocess data before feature generation
        processed_df = self._preprocess_data(df, column_mapping, engine_type)
        
        # Route to appropriate processor
        if engine_type == "paired_row":
            return self.paired_row_processor.generate_historical_features(processed_df, column_mapping, config)
        elif engine_type == "single_row":
            return self.single_row_processor.generate_historical_features(processed_df, column_mapping, config)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
    
    def _preprocess_data(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                        engine_type: str) -> pd.DataFrame:
        """Common preprocessing steps for both engines."""
        df_processed = df.copy()
        
        # Convert date column
        date_col = column_mapping.get('date')
        if date_col and date_col in df_processed.columns:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
            df_processed = df_processed.dropna(subset=[date_col])
        
        # Convert numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
        # Add Winner column for paired participant data
        if engine_type == "paired_row":
            df_processed = self._add_winner_column(df_processed, column_mapping)
        
        # Sort by date and event
        if date_col:
            event_id_col = column_mapping.get('event_id')
            if event_id_col:
                df_processed = df_processed.sort_values([date_col, event_id_col])
            else:
                df_processed = df_processed.sort_values([date_col])
        
        return df_processed.reset_index(drop=True)
    
    def _add_winner_column(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Add Winner column for paired participant data."""
        winner_first_col = column_mapping.get('winner_first')
        winner_last_col = column_mapping.get('winner_last')
        participant_col = column_mapping.get('participant_name')
        
        if all([winner_first_col, winner_last_col, participant_col]):
            df['Winner_Full_Name'] = (
                df[winner_first_col].astype(str) + ' ' + 
                df[winner_last_col].astype(str)
            ).str.upper().str.strip()
            
            df['Winner'] = (
                df[participant_col].astype(str).str.upper().str.strip() == 
                df['Winner_Full_Name']
            )
        elif winner_first_col and participant_col:
            # Use only first name if last name not available - iterate row by row
            def check_winner(row):
                participant = str(row[participant_col]).upper().strip()
                winner_first = str(row[winner_first_col]).upper().strip()
                return winner_first in participant
            
            df['Winner'] = df.apply(check_winner, axis=1)
        
        return df


# ============================================================================
# STREAMLIT USER INTERFACE
# ============================================================================

# --- Caching Functions ---
@st.cache_data
def process_sports_data(df: pd.DataFrame, column_mapping: Dict[str, str], 
                       detection_result: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Main cached function for processing sports data."""
    processor = UniversalSportsProcessor()
    return processor.process_data(df, column_mapping, detection_result, config)

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Universal Sports ML Pipeline",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Title and Description ---
st.title('🏆 Universal Sports ML Feature Engineering Pipeline')

st.markdown("""
This advanced application transforms raw sports data into **machine learning-ready datasets** with sophisticated historical features. 
It intelligently adapts to **any sport** - from Fighting/MMA to Football, Cricket, Basketball, and more!

### 🎯 Key Features
- **🧠 Intelligent Sport Detection**: Automatically identifies sport type and data structure
- **🗺️ Universal Column Mapping**: Smart auto-mapping with manual override capabilities  
- **📊 Advanced Feature Engineering**: Historical statistics with customizable lookback windows
- **⚡ High Performance**: Optimized processing with caching for large datasets
- **🎨 Intuitive Interface**: User-friendly design with real-time feedback
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "📁 Upload Sports Dataset", 
        type=["csv", "xlsx"],
        help="Upload your raw sports data file (CSV or Excel format)"
    )
    
    if uploaded_file:
        st.success("✅ File loaded successfully!")
    
    st.divider()
    
    # Feature Engineering Controls
    st.header("🔧 Feature Engineering")
    
    aggregation_types = st.multiselect(
        "📈 Aggregation Methods",
        ['Average', 'Sum', 'Median'],
        default=['Average'],
        help="Statistical methods to apply to historical data"
    )
    
    lookback_input = st.text_input(
        "🔢 Lookback Windows (comma-separated)",
        value="0, 5, 10",
        help="Number of previous events to consider (0 = all history)"
    )
    
    try:
        lookback_windows = [int(x.strip()) for x in lookback_input.split(',') if x.strip()]
    except ValueError:
        st.error("⚠️ Please enter valid numbers separated by commas")
        lookback_windows = [0, 5, 10]
    
    # Combat Sports Specific Options (Dynamic based on detection)
    st.subheader("🥊 Paired-Row Engine Options")
    
    # Check if we have detection result to show/hide options
    detection_result = st.session_state.get('detection_result')
    show_paired_options = detection_result and detection_result.get('engine') == 'paired_row'
    
    if show_paired_options:
        outcome_types = st.multiselect(
            "🏆 Outcome Types",
            ['Wins', 'Losses'],
            default=['Wins', 'Losses'],
            help="Generate separate stats for wins and losses"
        )
        
        shuffle_perspectives = st.checkbox(
            "🔀 Shuffle Participant Perspectives",
            help="Randomize which participant is 'A' vs 'B' in the final output"
        )
    else:
        outcome_types = ['Wins', 'Losses']  # Default values
        shuffle_perspectives = False
        st.info("ℹ️ Options will appear when Paired-Row engine is detected")
    
    # Betting Analysis (Dynamic based on odds mapping)
    st.subheader("💰 Betting Analysis")
    
    # This will be updated dynamically based on odds column mapping
    has_odds = st.session_state.get('has_odds_column', False)
    
    if has_odds:
        betting_statuses = st.multiselect(
            "📊 Betting Status Analysis",
            ['None', 'Overall', 'As Favorite', 'As Underdog'],
            default=['Overall'],
            help="Choose betting analysis options. Select 'None' to skip favorite/underdog calculations"
        )
    else:
        betting_statuses = ['Overall']
        st.info("ℹ️ Betting analysis options will appear when odds column is mapped")

# --- Main Application Logic ---
if 'training_df' not in st.session_state:
    st.session_state.training_df = None
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None

if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("📖 Reading uploaded file..."):
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                raw_df = pd.read_excel(uploaded_file)
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Rows", f"{len(raw_df):,}")
        with col2:
            st.metric("📋 Total Columns", len(raw_df.columns))
        with col3:
            st.metric("🔢 Numeric Columns", len(raw_df.select_dtypes(include=[np.number]).columns))
        
        # Data Preview
        with st.expander("👀 Raw Data Preview", expanded=True):
            st.write(f"**Showing {len(raw_df):,} rows and {len(raw_df.columns)} columns**")
            st.dataframe(
                raw_df, 
                use_container_width=True,
                height=400
            )
        
        st.divider()
        
        # Column Mapping Section
        st.header("🗺️ Intelligent Column Mapping")
        
        # Auto-detect columns
        auto_mapping = auto_map_columns(raw_df.columns.tolist())
        columns_list = ['None'] + raw_df.columns.tolist()
        
        # Show detected mappings
        if auto_mapping:
            st.success("✅ **Auto-detected mappings:**")
            detected_cols = st.columns(len(auto_mapping))
            for idx, (key, value) in enumerate(auto_mapping.items()):
                with detected_cols[idx % len(detected_cols)]:
                    st.write(f"**{key}**: {value}")
        
        # Manual mapping interface
        st.subheader("📝 Review and Adjust Mappings")
        
        # Universal mapping inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🆔 Core Identifiers**")
            event_id = st.selectbox(
                "Event/Match ID",
                columns_list,
                index=columns_list.index(auto_mapping['event_id']) if auto_mapping.get('event_id') and auto_mapping['event_id'] in columns_list else 0,
                help="Unique identifier for each event/match"
            )
            
            date_col = st.selectbox(
                "Event Date",
                columns_list,
                index=columns_list.index(auto_mapping['date']) if auto_mapping.get('date') and auto_mapping['date'] in columns_list else 0,
                help="Date when the event occurred"
            )
            
            # Enhanced Odds Section
            st.markdown("**💰 Betting Odds Configuration**")
            odds_type = st.radio(
                "Odds Structure",
                ["Single Odds (Combat Sports)", "Multi-Odds (Team Sports)", "No Odds"],
                help="Choose the odds structure in your dataset"
            )
            
            if odds_type == "Single Odds (Combat Sports)":
                odds_col = st.selectbox(
                    "Odds Column",
                    columns_list,
                    index=columns_list.index(auto_mapping['odds']) if auto_mapping.get('odds') and auto_mapping['odds'] in columns_list else 0,
                    help="Single odds column for combat sports"
                )
                odds_home_col = None
                odds_away_col = None
                odds_draw_col = None
                
            elif odds_type == "Multi-Odds (Team Sports)":
                odds_col = None
                st.markdown("*Select odds columns for each outcome:*")
                
                odds_home_col = st.selectbox(
                    "Home Win Odds",
                    columns_list,
                    index=columns_list.index(auto_mapping['odds_home']) if auto_mapping.get('odds_home') and auto_mapping['odds_home'] in columns_list else 0,
                    help="Odds for home team winning"
                )
                
                odds_away_col = st.selectbox(
                    "Away Win Odds",
                    columns_list,
                    index=columns_list.index(auto_mapping['odds_away']) if auto_mapping.get('odds_away') and auto_mapping['odds_away'] in columns_list else 0,
                    help="Odds for away team winning"
                )
                
                odds_draw_col = st.selectbox(
                    "Draw Odds",
                    columns_list,
                    index=columns_list.index(auto_mapping['odds_draw']) if auto_mapping.get('odds_draw') and auto_mapping['odds_draw'] in columns_list else 0,
                    help="Odds for draw/tie outcome"
                )
            else:
                odds_col = None
                odds_home_col = None
                odds_away_col = None
                odds_draw_col = None
        
        with col2:
            st.markdown("**⚔️ Individual Combat Sports**")
            participant_name = st.selectbox(
                "Participant Name",
                columns_list,
                index=columns_list.index(auto_mapping['participant_name']) if auto_mapping.get('participant_name') and auto_mapping['participant_name'] in columns_list else 0,
                help="Full name of individual participants (fighters, players)"
            )
            
            winner_first = st.selectbox(
                "Winner First Name",
                columns_list,
                index=columns_list.index(auto_mapping['winner_first']) if auto_mapping.get('winner_first') and auto_mapping['winner_first'] in columns_list else 0,
                help="First name of the event winner"
            )
            
            winner_last = st.selectbox(
                "Winner Last Name",
                columns_list,
                index=columns_list.index(auto_mapping['winner_last']) if auto_mapping.get('winner_last') and auto_mapping['winner_last'] in columns_list else 0,
                help="Last name of the event winner"
            )
        
        # Team Sports Mapping
        st.markdown("**🏈 Team Sports**")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            home_team = st.selectbox(
                "Home Team",
                columns_list,
                index=columns_list.index(auto_mapping['team1']) if auto_mapping.get('team1') and auto_mapping['team1'] in columns_list else 0,
                help="Home team or first team"
            )
        
        with col4:
            away_team = st.selectbox(
                "Away Team",
                columns_list,
                index=columns_list.index(auto_mapping['team2']) if auto_mapping.get('team2') and auto_mapping['team2'] in columns_list else 0,
                help="Away team or second team"
            )
        
        with col5:
            outcome_col = st.selectbox(
                "Match Outcome",
                columns_list,
                index=columns_list.index(auto_mapping['outcome']) if auto_mapping.get('outcome') and auto_mapping['outcome'] in columns_list else 0,
                help="Result of the match (H/A/D or similar)"
            )
        
        # Enhanced Team Sports Statistical Columns
        st.markdown("**📊 Team Sports Statistical Columns**")
        col6, col7 = st.columns(2)
        
        with col6:
            st.markdown("*Home Team Stats:*")
            score_home = st.selectbox(
                "Home Goals/Points", columns_list,
                index=columns_list.index(auto_mapping['score_home']) if auto_mapping.get('score_home') and auto_mapping['score_home'] in columns_list else 0,
                help="Goals/points scored by home team"
            )
            shots_home = st.selectbox(
                "Home Shots", columns_list,
                index=columns_list.index(auto_mapping['shots_home']) if auto_mapping.get('shots_home') and auto_mapping['shots_home'] in columns_list else 0,
                help="Total shots by home team"
            )
            shots_target_home = st.selectbox(
                "Home Shots on Target", columns_list,
                index=columns_list.index(auto_mapping['shots_target_home']) if auto_mapping.get('shots_target_home') and auto_mapping['shots_target_home'] in columns_list else 0,
                help="Shots on target by home team"
            )
            fouls_home = st.selectbox(
                "Home Fouls", columns_list,
                index=columns_list.index(auto_mapping['fouls_home']) if auto_mapping.get('fouls_home') and auto_mapping['fouls_home'] in columns_list else 0,
                help="Fouls committed by home team"
            )
            corners_home = st.selectbox(
                "Home Corners", columns_list,
                index=columns_list.index(auto_mapping['corners_home']) if auto_mapping.get('corners_home') and auto_mapping['corners_home'] in columns_list else 0,
                help="Corner kicks by home team"
            )
            yellows_home = st.selectbox(
                "Home Yellow Cards", columns_list,
                index=columns_list.index(auto_mapping['yellows_home']) if auto_mapping.get('yellows_home') and auto_mapping['yellows_home'] in columns_list else 0,
                help="Yellow cards for home team"
            )
            reds_home = st.selectbox(
                "Home Red Cards", columns_list,
                index=columns_list.index(auto_mapping['reds_home']) if auto_mapping.get('reds_home') and auto_mapping['reds_home'] in columns_list else 0,
                help="Red cards for home team"
            )
        
        with col7:
            st.markdown("*Away Team Stats:*")
            score_away = st.selectbox(
                "Away Goals/Points", columns_list,
                index=columns_list.index(auto_mapping['score_away']) if auto_mapping.get('score_away') and auto_mapping['score_away'] in columns_list else 0,
                help="Goals/points scored by away team"
            )
            shots_away = st.selectbox(
                "Away Shots", columns_list,
                index=columns_list.index(auto_mapping['shots_away']) if auto_mapping.get('shots_away') and auto_mapping['shots_away'] in columns_list else 0,
                help="Total shots by away team"
            )
            shots_target_away = st.selectbox(
                "Away Shots on Target", columns_list,
                index=columns_list.index(auto_mapping['shots_target_away']) if auto_mapping.get('shots_target_away') and auto_mapping['shots_target_away'] in columns_list else 0,
                help="Shots on target by away team"
            )
            fouls_away = st.selectbox(
                "Away Fouls", columns_list,
                index=columns_list.index(auto_mapping['fouls_away']) if auto_mapping.get('fouls_away') and auto_mapping['fouls_away'] in columns_list else 0,
                help="Fouls committed by away team"
            )
            corners_away = st.selectbox(
                "Away Corners", columns_list,
                index=columns_list.index(auto_mapping['corners_away']) if auto_mapping.get('corners_away') and auto_mapping['corners_away'] in columns_list else 0,
                help="Corner kicks by away team"
            )
            yellows_away = st.selectbox(
                "Away Yellow Cards", columns_list,
                index=columns_list.index(auto_mapping['yellows_away']) if auto_mapping.get('yellows_away') and auto_mapping['yellows_away'] in columns_list else 0,
                help="Yellow cards for away team"
            )
            reds_away = st.selectbox(
                "Away Red Cards", columns_list,
                index=columns_list.index(auto_mapping['reds_away']) if auto_mapping.get('reds_away') and auto_mapping['reds_away'] in columns_list else 0,
                help="Red cards for away team"
            )
        
        # Statistical Columns Selection
        st.subheader("📊 Statistical Features Selection")
        
        numerical_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        non_numerical_cols = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📈 **{len(numerical_cols)} Numerical Columns** available for feature engineering")
        with col_info2:
            st.info(f"📝 **{len(non_numerical_cols)} Non-Numerical Columns** will be preserved as-is")
        
        # Column selection with enhanced UI
        col_select1, col_select2 = st.columns([3, 1])
        
        with col_select1:
            stat_columns = st.multiselect(
                "Select columns for historical feature generation:",
                numerical_cols,
                default=numerical_cols,
                help="Choose which numerical columns to use for calculating historical statistics"
            )
        
        with col_select2:
            st.write("") # Spacing
            if st.button("🎯 Select All", use_container_width=True):
                stat_columns = numerical_cols
                st.rerun()
            
            if st.button("🧹 Clear All", use_container_width=True):
                stat_columns = []
                st.rerun()
        
        # Show selection summary
        if stat_columns:
            st.success(f"✅ **{len(stat_columns)} columns selected** for feature engineering")
            with st.expander("📋 Selected Columns"):
                cols = st.columns(4)
                for idx, col in enumerate(stat_columns):
                    with cols[idx % 4]:
                        st.write(f"• {col}")
        else:
            st.warning("⚠️ No statistical columns selected")
        
        # Preserve-as-is columns selection
        st.subheader("🔒 Preserve-as-is Columns")
        st.info("Select numerical columns to preserve without statistical calculations (e.g., IDs, dates, times)")
        
        # Identify likely preserve-as-is columns automatically
        likely_preserve_cols = []
        for col in numerical_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['id', 'time', 'date', 'year', 'month', 'day']):
                likely_preserve_cols.append(col)
        
        preserve_columns = st.multiselect(
            "Select columns to preserve as-is (won't be used for statistical calculations):",
            numerical_cols,
            default=likely_preserve_cols,
            help="These columns will be preserved in the output but won't have historical statistics calculated"
        )
        
        # Remove preserve columns from stat columns to avoid overlap
        stat_columns = [col for col in stat_columns if col not in preserve_columns]
        
        if preserve_columns:
            st.info(f"🔒 **{len(preserve_columns)} columns** will be preserved as-is")
        
        # Updated summary
        if stat_columns:
            st.success(f"✅ **{len(stat_columns)} columns** will be used for statistical feature engineering")
        else:
            st.warning("⚠️ No columns selected for statistical calculations")
        
        # Build column mapping - filter out None values
        column_mapping_raw = {
            'event_id': event_id if event_id != 'None' else None,
            'date': date_col if date_col != 'None' else None,
            'participant_name': participant_name if participant_name != 'None' else None,
            'team1': home_team if home_team != 'None' else None,
            'team2': away_team if away_team != 'None' else None,
            'outcome': outcome_col if outcome_col != 'None' else None,
            'odds': odds_col if odds_col != 'None' else None,
            'odds_home': odds_home_col if 'odds_home_col' in locals() and odds_home_col != 'None' else None,
            'odds_away': odds_away_col if 'odds_away_col' in locals() and odds_away_col != 'None' else None,
            'odds_draw': odds_draw_col if 'odds_draw_col' in locals() and odds_draw_col != 'None' else None,
            'winner_first': winner_first if winner_first != 'None' else None,
            'winner_last': winner_last if winner_last != 'None' else None,
            'score_home': score_home if 'score_home' in locals() and score_home != 'None' else None,
            'score_away': score_away if 'score_away' in locals() and score_away != 'None' else None,
            'shots_home': shots_home if 'shots_home' in locals() and shots_home != 'None' else None,
            'shots_away': shots_away if 'shots_away' in locals() and shots_away != 'None' else None,
            'shots_target_home': shots_target_home if 'shots_target_home' in locals() and shots_target_home != 'None' else None,
            'shots_target_away': shots_target_away if 'shots_target_away' in locals() and shots_target_away != 'None' else None,
            'fouls_home': fouls_home if 'fouls_home' in locals() and fouls_home != 'None' else None,
            'fouls_away': fouls_away if 'fouls_away' in locals() and fouls_away != 'None' else None,
            'corners_home': corners_home if 'corners_home' in locals() and corners_home != 'None' else None,
            'corners_away': corners_away if 'corners_away' in locals() and corners_away != 'None' else None,
            'yellows_home': yellows_home if 'yellows_home' in locals() and yellows_home != 'None' else None,
            'yellows_away': yellows_away if 'yellows_away' in locals() and yellows_away != 'None' else None,
            'reds_home': reds_home if 'reds_home' in locals() and reds_home != 'None' else None,
            'reds_away': reds_away if 'reds_away' in locals() and reds_away != 'None' else None
        }
        
        # Filter out None values for processing
        column_mapping = {k: v for k, v in column_mapping_raw.items() if v is not None}
        
        # Update sidebar based on current mappings
        has_odds = any([odds_col != 'None', 
                       'odds_home_col' in locals() and odds_home_col != 'None',
                       'odds_away_col' in locals() and odds_away_col != 'None', 
                       'odds_draw_col' in locals() and odds_draw_col != 'None'])
        st.session_state.has_odds_column = has_odds
        
        # Detect processing engine automatically
        detection_result = DataStructureDetector.detect_processing_engine(column_mapping)
        st.session_state.detection_result = detection_result
        
        # Validate engine requirements
        is_valid, validation_errors = DataStructureDetector.validate_engine_requirements(
            column_mapping, detection_result["engine"]
        )
        
        # Display detection results
        st.divider()
        st.header("🔍 Automatic Engine Detection")
        
        col_detect1, col_detect2, col_detect3 = st.columns(3)
        
        with col_detect1:
            engine_type = detection_result["engine"]
            if engine_type == "paired_row":
                st.success("🥊 **Paired-Row Engine**")
                st.write("Individual participants (2 rows per event)")
                st.write("Examples: MMA, Boxing, Tennis")
            elif engine_type == "single_row":
                st.success("🏈 **Single-Row Engine**")
                st.write("Team-based matches (1 row per event)")
                st.write("Examples: Football, Basketball, Cricket")
            else:
                st.error("❓ **Engine Unknown**")
                st.write("Cannot determine processing method")
        
        with col_detect2:
            structure = detection_result["structure"]
            confidence = detection_result["confidence"]
            
            if confidence >= 0.8:
                st.success(f"✅ **Confidence**: {confidence:.1%}")
            elif confidence >= 0.5:
                st.warning(f"⚠️ **Confidence**: {confidence:.1%}")
            else:
                st.error(f"❌ **Confidence**: {confidence:.1%}")
            
            st.write(f"**Structure**: {structure.replace('_', ' ').title()}")
        
        with col_detect3:
            trigger_fields = detection_result.get("trigger_fields", [])
            st.info("**🎯 Detection Triggers:**")
            if trigger_fields:
                for field in trigger_fields:
                    st.write(f"  ✓ {field.replace('_', ' ').title()}")
            else:
                st.write("  ❌ No triggers found")
        
        # Show validation results
        if not is_valid:
            st.error("❌ **Mapping Validation Failed**")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            st.success("✅ **Column mapping is valid!**")
        
        # Processing summary
        if stat_columns and is_valid:
            with st.expander("📋 Processing Summary"):
                st.write("**🔄 What will happen to your data:**")
                
                col_summary1, col_summary2 = st.columns(2)
                
                with col_summary1:
                    st.write("**📊 Selected Numerical Columns:**")
                    if stat_columns:
                        st.write("✅ **Will be processed for historical features:**")
                        for col in stat_columns[:5]:
                            st.write(f"  • {col} → avg, sum, median calculations")
                        if len(stat_columns) > 5:
                            st.write(f"  • ... and {len(stat_columns) - 5} more columns")
                    
                    remaining_numerical = [col for col in numerical_cols if col not in stat_columns]
                    if remaining_numerical:
                        st.write("🔒 **Will be preserved as-is:**")
                        for col in remaining_numerical[:3]:
                            st.write(f"  • {col}")
                        if len(remaining_numerical) > 3:
                            st.write(f"  • ... and {len(remaining_numerical) - 3} more")
                
                with col_summary2:
                    st.write("**📝 Non-Numerical Columns:**")
                    if non_numerical_cols:
                        st.write("🔒 **Will be preserved exactly as-is:**")
                        for col in non_numerical_cols[:5]:
                            st.write(f"  • {col} (preserved)")
                        if len(non_numerical_cols) > 5:
                            st.write(f"  • ... and {len(non_numerical_cols) - 5} more")
                    else:
                        st.write("ℹ️ All columns are numerical")
                    
                    st.write("**🆕 New Columns Created:**")
                    if detection_result.get("engine") == "paired_row":
                        st.write("  • Participant historical stats")
                        st.write("  • Opponent historical stats")
                        st.write("  • Difference features (participant - opponent)")
                        st.write("  • Betting odds analysis (if available)")
                    elif detection_result.get("engine") == "single_row":
                        st.write("  • Home team historical stats")
                        st.write("  • Away team historical stats") 
                        st.write("  • Difference features (home - away)")
                        st.write("  • Win/loss segmented analysis")
        
        # Generate Training Data Button
        st.divider()
        
        if st.button("🚀 **Generate ML-Ready Dataset**", 
                    type="primary", 
                    use_container_width=True,
                    disabled=not (stat_columns and is_valid)):
            
            if not stat_columns:
                st.error("❌ Please select at least one statistical column")
            elif not is_valid:
                st.error("❌ Please fix column mapping errors first")
            else:
                # Prepare configuration
                config = {
                    'aggregation_types': aggregation_types,
                    'lookback_windows': lookback_windows,
                    'outcome_types': outcome_types,
                    'betting_statuses': betting_statuses,
                    'stat_columns': stat_columns,
                    'preserve_columns': preserve_columns,
                    'odds_column': odds_col if odds_col != 'None' else None,
                    'shuffle_perspectives': shuffle_perspectives
                }
                
                # Process data
                with st.spinner("🔄 Generating advanced ML features... This may take a few moments."):
                    try:
                        st.session_state.training_df = process_sports_data(
                            raw_df, column_mapping, detection_result, config
                        )
                        st.success("🎉 **ML-ready dataset generated successfully!**")
                        
                        # Show quick stats
                        result_df = st.session_state.training_df
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        
                        with col_stats1:
                            st.metric("📊 Generated Rows", f"{len(result_df):,}")
                        with col_stats2:
                            st.metric("📋 Total Features", len(result_df.columns))
                        with col_stats3:
                            new_features = len(result_df.columns) - len(raw_df.columns)
                            st.metric("🆕 New Features", new_features)
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)
        
        # Display results
        if st.session_state.training_df is not None:
            st.divider()
            st.header("🎯 Generated ML-Ready Dataset")
            
            result_df = st.session_state.training_df
            
            # Dataset overview
            with st.expander("📈 Dataset Overview", expanded=True):
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("📊 Total Rows", f"{len(result_df):,}")
                with col_info2:
                    st.metric("📋 Total Columns", len(result_df.columns))
                with col_info3:
                    st.metric("🔢 Numeric Columns", len(result_df.select_dtypes(include=[np.number]).columns))
                
                st.write(f"**Full Dataset ({len(result_df):,} rows × {len(result_df.columns)} columns)**")
                st.dataframe(
                    result_df, 
                    use_container_width=True,
                    height=500
                )
            
            # Download options
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"ml_ready_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_download2:
                # Excel download
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    result_df.to_excel(writer, sheet_name='ML_Dataset', index=False)
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"ml_ready_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Feature analysis
            with st.expander("🔍 Feature Analysis"):
                feature_cols = [col for col in result_df.columns if any(
                    keyword in col.lower() for keyword in ['avg', 'sum', 'median', 'win', 'loss', 'diff']
                )]
                
                if feature_cols:
                    st.write(f"**🎯 {len(feature_cols)} engineered features created:**")
                    
                    # Group features by type
                    win_features = [col for col in feature_cols if 'win' in col.lower()]
                    diff_features = [col for col in feature_cols if 'diff' in col.lower()]
                    avg_features = [col for col in feature_cols if 'avg' in col.lower()]
                    
                    col_feat1, col_feat2, col_feat3 = st.columns(3)
                    
                    with col_feat1:
                        st.metric("🏆 Win-based Features", len(win_features))
                    with col_feat2:
                        st.metric("🔄 Difference Features", len(diff_features))
                    with col_feat3:
                        st.metric("📊 Average Features", len(avg_features))
            
            # ============================================================================
            # FILTERING & ANALYSIS SECTION
            # ============================================================================
            
            st.divider()
            st.header("🔍 Data Filtering & Analysis Tools")
            
            # Filtered View Section
            with st.expander("📊 Filtered View", expanded=False):
                st.subheader("Filter Generated Dataset")
                
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    # Date range filter
                    if 'date' in result_df.columns:
                        date_min = result_df['date'].min()
                        date_max = result_df['date'].max()
                        
                        date_range = st.date_input(
                            "📅 Date Range",
                            value=(date_min, date_max),
                            min_value=date_min,
                            max_value=date_max,
                            help="Filter results by date range"
                        )
                        
                        if len(date_range) == 2:
                            filtered_df = result_df[
                                (result_df['date'] >= pd.to_datetime(date_range[0])) &
                                (result_df['date'] <= pd.to_datetime(date_range[1]))
                            ]
                        else:
                            filtered_df = result_df
                    else:
                        filtered_df = result_df
                        st.info("No date column found for filtering")
                
                with filter_col2:
                    # Column selector for display
                    display_columns = st.multiselect(
                        "📋 Columns to Display",
                        result_df.columns.tolist(),
                        default=result_df.columns[:10].tolist(),
                        help="Choose which columns to show in the filtered view"
                    )
                
                # Show filtered results
                if display_columns:
                    st.write(f"**Showing {len(filtered_df):,} rows with {len(display_columns)} columns**")
                    st.dataframe(
                        filtered_df[display_columns], 
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.warning("Please select at least one column to display")
            
            # Participant Analysis Section
            with st.expander("👥 Participant Analysis", expanded=False):
                st.subheader("Head-to-Head & Performance Analysis")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    # Participant selector (dynamic based on engine type)
                    engine_type = detection_result.get("engine", "unknown")
                    
                    if engine_type == "paired_row":
                        # Individual participants
                        if 'participant_name' in result_df.columns:
                            unique_participants = result_df['participant_name'].unique()
                            selected_participant = st.selectbox(
                                "👤 Select Participant",
                                unique_participants,
                                help="Choose a participant to analyze their performance"
                            )
                            
                            participant_data = result_df[result_df['participant_name'] == selected_participant]
                            
                            # Show participant stats
                            st.write(f"**📊 {selected_participant} Performance:**")
                            if 'participant_won' in result_df.columns:
                                total_events = len(participant_data)
                                wins = participant_data['participant_won'].sum()
                                win_rate = wins / total_events if total_events > 0 else 0
                                
                                perf_col1, perf_col2, perf_col3 = st.columns(3)
                                with perf_col1:
                                    st.metric("Total Events", total_events)
                                with perf_col2:
                                    st.metric("Wins", int(wins))
                                with perf_col3:
                                    st.metric("Win Rate", f"{win_rate:.1%}")
                    
                    elif engine_type == "single_row":
                        # Team-based analysis
                        team_columns = [col for col in ['team1', 'team2'] if col in column_mapping_raw and column_mapping_raw[col] != 'None']
                        
                        if team_columns:
                            # Get unique teams from mapped columns
                            all_teams = set()
                            for col_key in team_columns:
                                actual_col = column_mapping_raw[col_key]
                                if actual_col in result_df.columns:
                                    all_teams.update(result_df[actual_col].unique())
                            
                            if all_teams:
                                selected_team = st.selectbox(
                                    "🏈 Select Team",
                                    sorted(all_teams),
                                    help="Choose a team to analyze their performance"
                                )
                                
                                # Find matches involving this team
                                team_matches = pd.DataFrame()
                                for col_key in team_columns:
                                    actual_col = column_mapping_raw[col_key]
                                    if actual_col in result_df.columns:
                                        matches = result_df[result_df[actual_col] == selected_team]
                                        team_matches = pd.concat([team_matches, matches]).drop_duplicates()
                                
                                st.write(f"**📊 {selected_team} Performance:**")
                                if not team_matches.empty:
                                    st.metric("Total Matches", len(team_matches))
                
                with analysis_col2:
                    # Feature comparison
                    st.write("**📈 Feature Comparison:**")
                    
                    # Get numerical features for comparison
                    numerical_features = [col for col in result_df.columns if 
                                        result_df[col].dtype in ['int64', 'float64'] and
                                        any(keyword in col.lower() for keyword in ['avg', 'sum', 'median', 'rate'])]
                    
                    if numerical_features:
                        selected_feature = st.selectbox(
                            "📊 Compare Feature",
                            numerical_features[:10],  # Limit to first 10 for performance
                            help="Choose a feature to compare across participants/teams"
                        )
                        
                        if selected_feature in result_df.columns:
                            feature_stats = result_df[selected_feature].describe()
                            
                            stat_col1, stat_col2 = st.columns(2)
                            with stat_col1:
                                st.metric("Average", f"{feature_stats['mean']:.2f}")
                                st.metric("Median", f"{feature_stats['50%']:.2f}")
                            with stat_col2:
                                st.metric("Max", f"{feature_stats['max']:.2f}")
                                st.metric("Min", f"{feature_stats['min']:.2f}")
                    else:
                        st.info("No numerical features available for comparison")
                        
                        
    except Exception as e:
        st.error(f"❌ **Error loading file**: {str(e)}")
        st.exception(e)

else:
    # Welcome screen
    st.info("👆 **Get started by uploading your sports dataset using the sidebar**")
    
    # Example datasets info
    with st.expander("📚 Supported Dataset Types & Examples"):
        st.markdown("""
        ### 🥊 **Individual Combat Sports**
        - **MMA/UFC**: Fighter stats, fight outcomes, betting odds
        - **Boxing**: Boxer records, round-by-round data
        - **Tennis**: Player stats, match results, tournament data
        
        ### 🏈 **Team Sports**
        - **Football/Soccer**: Team stats, match results, league data
        - **Basketball**: Team performance, game statistics
        - **Cricket**: Team scores, match outcomes, tournament data
        - **American Football**: Team stats, game results
        
        ### 📊 **Required Columns**
        
        **For Individual Sports:**
        - ✅ Participant name (fighter, player)
        - ✅ Event ID (fight_id, match_id)
        - ✅ Date (event_date, match_date)
        - ✅ Winner information (winner_first_name, winner_last_name)
        - 📊 Statistical columns (strikes, takedowns, etc.)
        
        **For Team Sports:**
        - ✅ Home team name
        - ✅ Away team name  
        - ✅ Date (match_date, game_date)
        - ✅ Match outcome (H/A/D, Home/Away/Draw)
        - 📊 Statistical columns (goals, shots, possession, etc.)
        """)
    
    # Feature benefits
    with st.expander("✨ What Makes This Pipeline Special?"):
        st.markdown("""
        ### 🧠 **Intelligent Processing**
        - **Auto-Detection**: Automatically identifies sport type and data structure
        - **Smart Mapping**: Intelligent column mapping with manual override
        - **Flexible Lookbacks**: Configurable historical windows (last 5, 10, all-time)
        
        ### 📈 **Advanced Features**
        - **Historical Statistics**: Win rates, averages, sums, medians
        - **Opponent Analysis**: Head-to-head and opponent-specific metrics
        - **Betting Intelligence**: Favorite vs underdog performance analysis
        - **Difference Features**: Automatic calculation of performance gaps
        
        ### ⚡ **Performance & Usability**
        - **Optimized Processing**: Cached computations for large datasets
        - **Progress Tracking**: Real-time progress bars and status updates
        - **Export Options**: Download as CSV or Excel with one click
        - **Validation**: Built-in data validation and error checking
        """)
