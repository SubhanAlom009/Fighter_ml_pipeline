import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import random

# ---------------- UNIVERSAL SCHEMA ---------------- #
UNIVERSAL_SCHEMA = {
    "date": ["date", "match_date", "event_date"],
    "team1": ["hometeam", "team1", "home"],
    "team2": ["awayteam", "team2", "away"],
    "outcome": ["result", "winner", "ftr"],
    "odds": ["odds", "betting_line", "b365h", "psh"],
    "event_id": ["match_id", "event_id", "game_id", "fight id"],
    "participant_name": ["fighter full name", "player", "participant", "fighter", "athlete", "name"],
    "winner_first": ["winner first name", "winner_first", "winnerfirstname"],
    "winner_last": ["winner last name", "winner_last", "winnerlastname"]
}

def auto_map_columns(raw_columns, schema=UNIVERSAL_SCHEMA):
    """
    Automatically map raw dataset columns to universal schema keys.
    """
    mapping = {}
    for key, synonyms in schema.items():
        for col in raw_columns:
            # Simple substring matching for better flexibility
            if any(s in col.lower().replace("_", " ") for s in synonyms):
                mapping[key] = col
                break
    return mapping


# --- Caching Decorators for Optimization --- #
@st.cache_data
def preprocess_data(df, column_mapping):
    """
    Preprocesses the raw data based on its structure type and user mappings.
    """
    df_processed = df.copy()
    date_col = column_mapping.get('date')

    if date_col and date_col in df_processed.columns:
        df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')

    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

    # This logic is now primarily for Paired-Row data
    winner_first_name_col = column_mapping.get('winner_first')
    winner_last_name_col = column_mapping.get('winner_last')
    participant_name_col = column_mapping.get('participant_name')
    event_id_col = column_mapping.get('event_id')
    
    if all([winner_first_name_col, winner_last_name_col, participant_name_col]):
        df_processed['Winner Full Name'] = (df_processed[winner_first_name_col].astype(str) + ' ' + df_processed[winner_last_name_col].astype(str)).str.upper()
        df_processed['Winner'] = df_processed[participant_name_col].astype(str).str.upper() == df_processed['Winner Full Name']
    
    if date_col and event_id_col:
        df_processed = df_processed.sort_values(by=[date_col, event_id_col]).reset_index(drop=True)
    elif date_col:
         df_processed = df_processed.sort_values(by=[date_col]).reset_index(drop=True)

    return df_processed


def calculate_historical_stats_paired(fighter_df, aggregation_types, last_n_fights_list, outcome_types, betting_statuses, stat_columns, odds_col):
    all_stats = {}
    for last_n in last_n_fights_list:
        window_df = fighter_df.copy()
        if last_n > 0:
            window_df = window_df.tail(last_n)

        wins = window_df[window_df['Winner']]
        losses = window_df[~window_df['Winner']]
        all_stats[f'NumberOf_Fight_{last_n}'] = len(window_df)
        all_stats[f'NumberOf_WIN_{last_n}'] = len(wins)
        all_stats[f'NumberOf_LOSE_{last_n}'] = len(losses)
        all_stats[f'WIN_RATE_{last_n}'] = all_stats[f'NumberOf_WIN_{last_n}'] / all_stats[f'NumberOf_Fight_{last_n}'] if all_stats[f'NumberOf_Fight_{last_n}'] > 0 else 0

        segments = {}
        if 'Overall' in betting_statuses:
            if 'Wins' in outcome_types: segments[''] = wins
            if 'Losses' in outcome_types: segments['lose_'] = losses
        
        if odds_col and odds_col in window_df.columns:
            if 'As Favorite' in betting_statuses:
                favorite_fights = window_df[window_df[odds_col] < 0]
                if 'Wins' in outcome_types: segments['_favorite'] = favorite_fights[favorite_fights['Winner']]
                if 'Losses' in outcome_types: segments['lose__favorite'] = favorite_fights[~favorite_fights['Winner']]
            if 'As Underdog' in betting_statuses:
                underdog_fights = window_df[window_df[odds_col] >= 0]
                if 'Wins' in outcome_types: segments['_underdog'] = underdog_fights[underdog_fights['Winner']]
                if 'Losses' in outcome_types: segments['lose__underdog'] = underdog_fights[~underdog_fights['Winner']]

        agg_map = {'Average': 'mean', 'Sum': 'sum', 'Median': 'median'}
        for agg_type in aggregation_types:
            agg_func_name = 'avg' if agg_type == 'Average' else agg_type.lower()
            agg_method = agg_map[agg_type]
            for col in stat_columns:
                for suffix, data_segment in segments.items():
                    prefix = 'lose' if 'lose_' in suffix else 'win'
                    clean_suffix = suffix.replace('lose_', '')
                    stat_name = f'{prefix}_{agg_func_name}_{col.replace(" ", "_")}{clean_suffix}_{last_n}'
                    all_stats[stat_name] = data_segment[col].agg(agg_method) if not data_segment.empty else 0
    return pd.Series(all_stats).fillna(0)

def calculate_historical_stats_single(participant_history_df, aggregation_types, last_n_events_list, stat_columns, home_team_col, away_team_col, ftr_col):
    all_stats = {}
    for last_n in last_n_events_list:
        window_df = participant_history_df.copy()
        if last_n > 0:
            window_df = window_df.tail(last_n)
        
        # Add a 'Winner' column to each history DataFrame
        def add_winner_column(history, team_name):
            history['Winner'] = np.where(
                ((history[home_team_col] == team_name) & (history[ftr_col] == 'H')) |
                ((history[away_team_col] == team_name) & (history[ftr_col] == 'A')),
                True, False
            )
            return history
        
        window_df = add_winner_column(window_df, window_df.iloc[0]['team_name_temp'])
        
        wins = window_df[window_df['Winner']]
        losses = window_df[~window_df['Winner']]
        all_stats[f'NumberOf_Games_{last_n}'] = len(window_df)
        all_stats[f'NumberOf_WINS_{last_n}'] = len(wins)
        all_stats[f'NumberOf_LOSSES_{last_n}'] = len(losses)
        all_stats[f'WIN_RATE_{last_n}'] = all_stats[f'NumberOf_WINS_{last_n}'] / all_stats[f'NumberOf_Games_{last_n}'] if all_stats[f'NumberOf_Games_{last_n}'] > 0 else 0
        agg_map = {'Average': 'mean', 'Sum': 'sum', 'Median': 'median'}
        for agg_type in aggregation_types:
            agg_func_name = 'avg' if agg_type == 'Average' else agg_type.lower()
            agg_method = agg_map[agg_type]
            for col in stat_columns:
                win_stat_name = f'win_{agg_func_name}_{col.replace(" ", "_")}_{last_n}'
                lose_stat_name = f'lose_{agg_func_name}_{col.replace(" ", "_")}_{last_n}'
                all_stats[win_stat_name] = wins[col].agg(agg_method) if not wins.empty else 0
                all_stats[lose_stat_name] = losses[col].agg(agg_method) if not losses.empty else 0
    return pd.Series(all_stats).fillna(0)


# --- Caching Decorators for Optimization ---
@st.cache_data
def create_training_data_paired_row(_df, aggregation_types, last_n_fights_list, shuffle_perspectives, outcome_types, betting_statuses, column_mapping, stat_columns):
    df = _df.copy()
    processed_fights = []
    progress_bar_placeholder = st.sidebar.empty()
    progress_bar_placeholder.progress(0)
    total_rows = len(df)
    
    fighter_full_name_col = column_mapping['participant_name']
    fight_id_col = column_mapping['event_id']
    date_col = column_mapping['date']
    odds_col = column_mapping.get('odds')

    for i, fighter_x_info in df.iterrows():
        fight_id = fighter_x_info[fight_id_col]
        fight_date = fighter_x_info[date_col]
        
        opponent_query = df[(df[fight_id_col] == fight_id) & (df.index != i)]
        if opponent_query.empty:
            continue
        fighter_y_info = opponent_query.iloc[0]

        fighter_x_history = df[(df[fighter_full_name_col] == fighter_x_info[fighter_full_name_col]) & (df[date_col] < fight_date)]
        fighter_y_history = df[(df[fighter_full_name_col] == fighter_y_info[fighter_full_name_col]) & (df[date_col] < fight_date)]
        
        fighter_x_stats = calculate_historical_stats_paired(fighter_x_history, aggregation_types, last_n_fights_list, outcome_types, betting_statuses, stat_columns, odds_col)
        fighter_y_stats = calculate_historical_stats_paired(fighter_y_history, aggregation_types, last_n_fights_list, outcome_types, betting_statuses, stat_columns, odds_col)
        
        odds_x, odds_y, diff_odds = 0, 0, 0
        if odds_col:
            raw_odds_x = fighter_x_info.get(odds_col, 0)
            raw_odds_y = fighter_y_info.get(odds_col, 0)
            odds_x = 10000 / abs(raw_odds_x) if raw_odds_x < 0 else raw_odds_x
            odds_y = 10000 / abs(raw_odds_y) if raw_odds_y < 0 else raw_odds_y
            odds_x, odds_y = int(round(odds_x)), int(round(odds_y))
            diff_odds = odds_x - odds_y

        diff_stats = (fighter_x_stats - fighter_y_stats).add_prefix('diff_')
        
        fighter_x_stats = fighter_x_stats.add_suffix('_x')
        fighter_y_stats = fighter_y_stats.add_suffix('_y')
        
        fight_info = pd.Series({
            'fight_id': fight_id, 'date': fight_date, 'fighter_x_name': fighter_x_info[fighter_full_name_col],
            'fighter_y_name': fighter_y_info[fighter_full_name_col], 'fighter_x_win': int(fighter_x_info.get('Winner', False))
        })
        fighter_x_data = pd.concat([fighter_x_stats, pd.Series({'odds_x': odds_x})])
        fighter_y_data = pd.concat([fighter_y_stats, pd.Series({'odds_y': odds_y})])
        diff_data = pd.concat([diff_stats, pd.Series({'diff_odds': diff_odds})])

        combined_stats = pd.concat([fight_info, fighter_x_data, fighter_y_data, diff_data])
        processed_fights.append(combined_stats)
        progress_bar_placeholder.progress((i + 1) / total_rows)
    
    progress_bar_placeholder.empty()
    final_df = pd.DataFrame(processed_fights).fillna(0)
    
    if shuffle_perspectives:
        shuffled_list = []
        for _, group in final_df.groupby('fight_id'):
            shuffled_list.append(group.sample(frac=1))
        final_df = pd.concat(shuffled_list).reset_index(drop=True)
    else:
        final_df = final_df.sort_values(by=['fight_id', 'fighter_x_name']).reset_index(drop=True)
    
    return final_df

@st.cache_data
def create_training_data_single_row(_df, aggregation_types, last_n_events_list, column_mapping, stat_columns):
    df = _df.copy()
    processed_matches = []
    progress_bar_placeholder = st.sidebar.empty()
    progress_bar_placeholder.progress(0)
    total_rows = len(df)

    home_team_col = column_mapping['team1']
    away_team_col = column_mapping['team2']
    date_col = column_mapping['date']
    ftr_col = column_mapping['outcome']

    for i, match_info in df.iterrows():
        match_date = match_info[date_col]
        home_team = match_info[home_team_col]
        away_team = match_info[away_team_col]

        home_history_raw = df[((df[home_team_col] == home_team) | (df[away_team_col] == home_team)) & (df[date_col] < match_date)]
        away_history_raw = df[((df[away_team_col] == away_team) | (df[home_team_col] == away_team)) & (df[date_col] < match_date)]

        home_history_raw['team_name_temp'] = home_team
        away_history_raw['team_name_temp'] = away_team

        home_stats = calculate_historical_stats_single(home_history_raw, aggregation_types, last_n_events_list, stat_columns, home_team_col, away_team_col, ftr_col)
        away_stats = calculate_historical_stats_single(away_history_raw, aggregation_types, last_n_events_list, stat_columns, home_team_col, away_team_col, ftr_col)

        home_stats = home_stats.add_prefix('home_')
        away_stats = away_stats.add_prefix('away_')
        diff_stats = (home_stats.rename(lambda x: x.replace('home_', 'diff_')) - away_stats.rename(lambda x: x.replace('away_', 'diff_')))

        match_data = match_info.copy()
        combined_stats = pd.concat([match_data, home_stats, away_stats, diff_stats])
        processed_matches.append(combined_stats)
        progress_bar_placeholder.progress((i + 1) / total_rows)
    
    progress_bar_placeholder.empty()
    return pd.DataFrame(processed_matches).fillna(0)


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title('🏈 Universal Sports Data Feature Engineering 🥊')

st.markdown("""
This app transforms raw sports data into a "training-ready" dataset with advanced historical features. It intelligently adapts to your data's structure.
""")

with st.expander("ℹ️ How to Use This App & Feature Explanations"):
    st.markdown("""
    ### **How It Works**
    1.  **Upload Data**: Start by uploading your raw data file (`.csv` or `.xlsx`).
    2.  **Map Columns**: The app will auto-detect columns based on a universal schema. Review and adjust the mappings for your specific dataset. The app will automatically determine the data's structure (Paired-Row vs. Single-Row) based on your selections.
    3.  **Select Features**: Use the sidebar to choose the historical stats you want to generate.
    4.  **Generate Data**: Click the "Generate Training Data" button to run the appropriate pipeline.
    """)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Feature Engineering Controls")
    uploaded_file = st.file_uploader("Upload Raw Data File", type=["csv", "xlsx"])
    
    st.header("Feature Selection")
    aggregation_types = st.multiselect("Aggregation Types", ['Average', 'Sum', 'Median'], default=['Average'])
    last_n_events_input = st.text_input("'Last N Events' Windows (comma-separated)", value='0, 10')
    last_n_events_list = [int(x.strip()) for x in last_n_events_input.split(',')] if last_n_events_input else []
    
    # These will be shown for all, but only used by the paired-row engine
    shuffle_perspectives = st.checkbox("Shuffle Perspectives (for Paired-Row data)")
    outcome_types = st.multiselect("Outcome Types", ['Wins', 'Losses'], default=['Wins', 'Losses'])
    betting_statuses = st.multiselect("Betting Status (if odds provided)", ['Overall', 'As Favorite', 'As Underdog'], default=['Overall'])


# --- Main App Logic ---
if 'training_df' not in st.session_state:
    st.session_state.training_df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            raw_df = pd.read_excel(uploaded_file)
        
        st.success("✅ File uploaded and read successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head())
        
        st.subheader("🗺️ Column Mapping")
        st.info("The app will automatically determine the data structure based on your mappings.")
        columns = ['None'] + raw_df.columns.tolist()
        auto_mapping = auto_map_columns(raw_df.columns.tolist())
        
        # --- Unified Mapping UI ---
        st.markdown("##### Paired-Row Mapping (e.g., Fighting)")
        col1, col2, col3 = st.columns(3)
        with col1:
            participant_name = st.selectbox("Participant Full Name", columns, index=columns.index(auto_mapping['participant_name']) if 'participant_name' in auto_mapping else 0)
        with col2:
            winner_first = st.selectbox("Winner First Name", columns, index=columns.index(auto_mapping['winner_first']) if 'winner_first' in auto_mapping else 0)
        with col3:
            winner_last = st.selectbox("Winner Last Name", columns, index=columns.index(auto_mapping['winner_last']) if 'winner_last' in auto_mapping else 0)

        st.markdown("##### Single-Row Mapping (e.g., Team Sports)")
        col1, col2, col3 = st.columns(3)
        with col1:
            home_team = st.selectbox("Home Participant", columns, index=columns.index(auto_mapping['team1']) if 'team1' in auto_mapping else 0)
        with col2:
            away_team = st.selectbox("Away Participant", columns, index=columns.index(auto_mapping['team2']) if 'team2' in auto_mapping else 0)
        with col3:
            result_col = st.selectbox("Result Column (e.g., FTR)", columns, index=columns.index(auto_mapping['outcome']) if 'outcome' in auto_mapping else 0)

        st.markdown("##### Common Columns")
        col1, col2 = st.columns(2)
        with col1:
            event_id = st.selectbox("Event ID", columns, index=columns.index(auto_mapping['event_id']) if 'event_id' in auto_mapping else 0)
        with col2:
            date_col = st.selectbox("Event Date", columns, index=columns.index(auto_mapping['date']) if 'date' in auto_mapping else 0)
        
        odds_col = st.selectbox("Odds (Optional)", columns, index=columns.index(auto_mapping['odds']) if 'odds' in auto_mapping else 0)
        stat_columns = st.multiselect("Select Statistical Columns to Process", raw_df.select_dtypes(include=np.number).columns.tolist(), default=raw_df.select_dtypes(include=np.number).columns.tolist())
        
        # --- Determine Structure and Build Mapping ---
        structure_type = None
        column_mapping = {} # FIX: Initialize to prevent UnboundVariable error
        if participant_name != 'None' and winner_first != 'None':
            structure_type = 'Paired-Row Data (e.g., Fighting)'
            column_mapping = {"participant_name": participant_name, "event_id": event_id, "date": date_col, "winner_first": winner_first, "winner_last": winner_last, "odds": odds_col}
        elif home_team != 'None' and away_team != 'None':
            structure_type = 'Single-Row Data (e.g., Team Sports)'
            column_mapping = {"team1": home_team, "team2": away_team, "date": date_col, "outcome": result_col}
        
        if structure_type:
            st.success(f"Detected Structure: **{structure_type}**")

        if st.button("🚀 Generate Training Data"):
            if not structure_type:
                st.error("Could not determine data structure. Please map the required columns for either a Paired-Row or Single-Row dataset.")
            else:
                processed_df = preprocess_data(raw_df, column_mapping)
                with st.spinner("Generating features..."):
                    if structure_type == 'Paired-Row Data (e.g., Fighting)':
                        st.session_state.training_df = create_training_data_paired_row(processed_df, tuple(aggregation_types), tuple(last_n_events_list), shuffle_perspectives, tuple(outcome_types), tuple(betting_statuses), column_mapping, tuple(stat_columns))
                    else: # Single-Row
                        st.session_state.training_df = create_training_data_single_row(processed_df, tuple(aggregation_types), tuple(last_n_events_list), column_mapping, tuple(stat_columns))
                st.success("🎉 Training data generated successfully!")

        if st.session_state.training_df is not None:
            st.subheader("Generated Training Data Preview")
            st.dataframe(st.session_state.training_df)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting for a CSV or XLSX file to be uploaded.")
