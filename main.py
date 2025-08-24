import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import random

# --- Caching Decorators for Optimization ---
@st.cache_data
def preprocess_raw_data(df):
    """
    Preprocesses the raw fighter data.
    - Converts date columns to datetime objects.
    - Cleans and fills null values for all critical numeric columns.
    - Creates a 'Winner' column to identify the winner of each fight.
    - Sorts data by date and then Fight ID to ensure a stable chronological order.
    """
    # Dynamically find date-like columns and convert them
    for col in df.columns:
        if 'date' in col.lower() or 'dob' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Identify numeric columns dynamically, excluding IDs
    id_cols = {'fight id', 'fighter id', 'match id'}
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col.lower() not in id_cols]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Heuristically find stance column
    stance_col = next((c for c in df.columns if 'stance' in c.lower()), None)
    if stance_col:
        df[stance_col] = df[stance_col].fillna('Unknown')

    # Dynamically find name columns to create Winner
    winner_first_name_col = next((c for c in df.columns if 'winner' in c.lower() and 'first' in c.lower()), None)
    winner_last_name_col = next((c for c in df.columns if 'winner' in c.lower() and 'last' in c.lower()), None)
    fighter_full_name_col = next((c for c in df.columns if ('fighter' in c.lower() or 'player' in c.lower()) and 'full' in c.lower()), None)

    if winner_first_name_col and winner_last_name_col and fighter_full_name_col:
        df['Winner Full Name'] = (df[winner_first_name_col].astype(str) + ' ' + df[winner_last_name_col].astype(str)).str.upper()
        df['Winner'] = df[fighter_full_name_col].str.upper() == df['Winner Full Name']
    
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    fight_id_col = next((c for c in df.columns if 'fight id' in c.lower() or 'match id' in c.lower()), None)
    
    # FIX: Ensure both date and fight ID columns exist before sorting
    if date_col and fight_id_col:
        df = df.sort_values(by=[date_col, fight_id_col]).reset_index(drop=True)
    return df

def calculate_historical_stats(fighter_df, aggregation_types, last_n_fights_list, outcome_types, betting_statuses):
    """
    Calculates historical stats for a single fighter's past performances for multiple lookback windows and user-selected criteria.
    """
    all_stats = {}
    
    # Dynamically identify statistical columns to aggregate
    core_cols = {'fight id', 'match id', 'date', 'fighter id', 'winner', 'odds'}
    stat_columns = [col for col in fighter_df.select_dtypes(include=np.number).columns if col.lower() not in core_cols and 'feet' not in col.lower() and 'inches' not in col.lower()]

    for last_n in last_n_fights_list:
        window_df = fighter_df.copy()
        if last_n > 0:
            window_df = window_df.tail(last_n)

        # --- Basic Career Stats ---
        wins = window_df[window_df['Winner']]
        losses = window_df[~window_df['Winner']]
        all_stats[f'NumberOf_Fight_{last_n}'] = len(window_df)
        all_stats[f'NumberOf_WIN_{last_n}'] = len(wins)
        all_stats[f'NumberOf_LOSE_{last_n}'] = len(losses)
        all_stats[f'WIN_RATE_{last_n}'] = all_stats[f'NumberOf_WIN_{last_n}'] / all_stats[f'NumberOf_Fight_{last_n}'] if all_stats[f'NumberOf_Fight_{last_n}'] > 0 else 0

        # --- Segment fights by favorite/underdog status ---
        favorite_fights = window_df[window_df['odds'] < 0]
        underdog_fights = window_df[window_df['odds'] >= 0]
        
        segments = {}
        if 'Overall' in betting_statuses:
            if 'Wins' in outcome_types: segments[''] = wins
            if 'Losses' in outcome_types: segments['lose_'] = losses
        if 'As Favorite' in betting_statuses:
            if 'Wins' in outcome_types: segments['_favorite'] = favorite_fights[favorite_fights['Winner']]
            if 'Losses' in outcome_types: segments['lose__favorite'] = favorite_fights[~favorite_fights['Winner']]
        if 'As Underdog' in betting_statuses:
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


# --- Caching Decorators for Optimization ---
@st.cache_data
def create_training_data(_df, aggregation_types, last_n_fights_list, shuffle_perspectives, outcome_types, betting_statuses):
    """
    Transforms the raw fight data into a training-ready format with organized columns.
    """
    df = _df.copy()
    processed_fights = []
    
    progress_bar_placeholder = st.sidebar.empty()
    progress_bar_placeholder.progress(0)
    total_rows = len(df)
    
    fighter_full_name_col = next((c for c in df.columns if ('fighter' in c.lower() or 'player' in c.lower()) and 'full' in c.lower()), None)
    fight_id_col = next((c for c in df.columns if 'fight id' in c.lower() or 'match id' in c.lower()), None)
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    dob_col = next((c for c in df.columns if 'dob' in c.lower() or 'date of birth' in c.lower()), None)
    height_feet_col = next((c for c in df.columns if 'height' in c.lower() and 'feet' in c.lower()), None)
    height_inches_col = next((c for c in df.columns if 'height' in c.lower() and 'inches' in c.lower()), None)
    weight_col = next((c for c in df.columns if 'weight' in c.lower()), None)
    reach_col = next((c for c in df.columns if 'reach' in c.lower()), None)
    stance_col = next((c for c in df.columns if 'stance' in c.lower()), None)


    for i, fighter_x_info in df.iterrows():
        fight_id = fighter_x_info[fight_id_col]
        fight_date = fighter_x_info[date_col]
        
        opponent_query = df[(df[fight_id_col] == fight_id) & (df.index != i)]
        if opponent_query.empty:
            continue
        fighter_y_info = opponent_query.iloc[0]

        fighter_x_history = df[(df[fighter_full_name_col] == fighter_x_info[fighter_full_name_col]) & (df[date_col] < fight_date)]
        fighter_y_history = df[(df[fighter_full_name_col] == fighter_y_info[fighter_full_name_col]) & (df[date_col] < fight_date)]
        
        fighter_x_stats = calculate_historical_stats(fighter_x_history, aggregation_types, last_n_fights_list, outcome_types, betting_statuses)
        fighter_y_stats = calculate_historical_stats(fighter_y_history, aggregation_types, last_n_fights_list, outcome_types, betting_statuses)
        
        static_features_x = {
            'Fighter ID_x': fight_id,
            'Age (in days)_x': (fight_date - fighter_x_info[dob_col]).days if dob_col and pd.notna(fighter_x_info.get(dob_col)) else 0,
            'Height_x': fighter_x_info.get(height_feet_col, 0) * 12 + fighter_x_info.get(height_inches_col, 0),
            'Weight Pounds_x': fighter_x_info.get(weight_col, 0),
            'Reach Inches_x': fighter_x_info.get(reach_col, 0),
            'Stance_x': fighter_x_info.get(stance_col, 'Unknown'),
        }
        static_features_y = {
            'Fighter ID_y': fight_id,
            'Age (in days)_y': (fight_date - fighter_y_info.get(dob_col)).days if dob_col and pd.notna(fighter_y_info.get(dob_col)) else 0,
            'Height_y': fighter_y_info.get(height_feet_col, 0) * 12 + fighter_y_info.get(height_inches_col, 0),
            'Weight Pounds_y': fighter_y_info.get(weight_col, 0),
            'Reach Inches_y': fighter_y_info.get(reach_col, 0),
            'Stance_y': fighter_y_info.get(stance_col, 'Unknown'),
        }

        raw_odds_x = fighter_x_info.get('odds', 0)
        raw_odds_y = fighter_y_info.get('odds', 0)
        odds_x = 10000 / abs(raw_odds_x) if raw_odds_x < 0 else raw_odds_x
        odds_y = 10000 / abs(raw_odds_y) if raw_odds_y < 0 else raw_odds_y
        odds_x, odds_y = int(round(odds_x)), int(round(odds_y))
        diff_odds = odds_x - odds_y

        diff_stats = (fighter_x_stats - fighter_y_stats).add_prefix('diff_')
        diff_static = {f"diff_{k.replace('_x','')}": v - static_features_y.get(k.replace('_x','_y'), 0) for k,v in static_features_x.items() if k != 'Stance_x'}
        
        fighter_x_stats = fighter_x_stats.add_suffix('_x')
        fighter_y_stats = fighter_y_stats.add_suffix('_y')
        
        fight_info = pd.Series({
            'fight_id': fight_id, 'date': fight_date, 'fighter_x_name': fighter_x_info[fighter_full_name_col],
            'fighter_y_name': fighter_y_info[fighter_full_name_col], 'fighter_x_win': int(fighter_x_info.get('Winner', False))
        })
        fighter_x_data = pd.concat([pd.Series(static_features_x), fighter_x_stats, pd.Series({'odds_x': odds_x})])
        fighter_y_data = pd.concat([pd.Series(static_features_y), fighter_y_stats, pd.Series({'odds_y': odds_y})])
        diff_data = pd.concat([pd.Series(diff_static), diff_stats, pd.Series({'diff_odds': diff_odds})])

        combined_stats = pd.concat([fight_info, fighter_x_data, fighter_y_data, diff_data])
        processed_fights.append(combined_stats)

        progress_bar_placeholder.progress((i + 1) / total_rows)
    
    progress_bar_placeholder.empty()
    final_df = pd.DataFrame(processed_fights).fillna(0)
    
    if shuffle_perspectives:
        shuffled_list = []
        for _, group in final_df.groupby(fight_id_col):
            shuffled_list.append(group.sample(frac=1))
        final_df = pd.concat(shuffled_list).reset_index(drop=True)
    else:
        # Only sort if fight_id_col was found, otherwise just sort by fighter name
        sort_cols = [fight_id_col, 'fighter_x_name'] if fight_id_col else ['fighter_x_name']
        final_df = final_df.sort_values(by=sort_cols).reset_index(drop=True)
    
    return final_df


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title('🥊 Universal Fight Data Feature Engineering')

st.markdown("""
This app transforms raw, fight-by-fight data into a "training-ready" dataset with advanced filtering and analysis. 
""")

with st.expander("ℹ️ How to Use This App & Feature Explanations"):
    st.markdown("""
    This tool is designed to be a flexible and powerful feature engineering pipeline for any fight sports dataset that follows a similar structure.

    ### **How It Works**
    1.  **Upload Data**: Start by uploading your raw data file (`.csv` or `.xlsx`). The app will automatically detect column types and prepare the data.
    2.  **Select Features**: Use the controls in the sidebar to choose exactly what kind of historical stats you want to generate.
        * **Outcome Types**: Choose whether to calculate stats for a fighter's `Wins`, `Losses`, or both.
        * **Betting Status**: Segment the stats based on whether the fighter was the betting `Favorite` (negative odds), `Underdog` (positive odds), or their `Overall` performance.
        * **Aggregation Types**: Select how you want to summarize the stats (`Average`, `Sum`, `Median`). You can select multiple types.
        * **'Last N Fights' Windows**: This is a powerful feature to analyze a fighter's form. You can enter multiple numbers (e.g., `0, 5, 10`). `0` calculates stats over their **entire career**, while `5` or `10` focuses on their **recent form**.
        * **Shuffle Perspectives**: A data augmentation feature. For each fight, the app creates two rows (one for each fighter's perspective). This option randomizes which fighter's perspective appears first for each fight pair, which can help prevent bias in machine learning models.
    3.  **Generate Data**: Click the "Generate Training Data" button to run the pipeline. The process is cached, so subsequent runs with the same settings will be instant.
    4.  **Analyze & Filter**: Use the "Filtered View" and "Fighter Analysis" sections to explore the generated dataset.
    """)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Feature Engineering Controls")
    
    uploaded_file = st.file_uploader("Upload Raw Data File", type=["csv", "xlsx"])
    
    st.header("Feature Selection")
    outcome_types = st.multiselect("Outcome Types", ['Wins', 'Losses'], default=['Wins', 'Losses'])
    betting_statuses = st.multiselect("Betting Status", ['Overall', 'As Favorite', 'As Underdog'], default=['Overall'])
    aggregation_types = st.multiselect("Aggregation Types", ['Average', 'Sum', 'Median'], default=['Average'])
    
    last_n_fights_input = st.text_input("'Last N Fights' Windows (comma-separated)", value='0, 10')
    last_n_fights_list = []
    if last_n_fights_input:
        try:
            last_n_fights_list = [int(x.strip()) for x in last_n_fights_input.split(',')]
        except ValueError:
            st.error("Invalid input for 'Last N Fights'. Please enter numbers separated by commas.")
            last_n_fights_list = []

    shuffle_perspectives = st.checkbox("Shuffle fighter perspectives (x vs y)", value=False)

# --- Main App Logic ---
if 'training_df' not in st.session_state:
    st.session_state.training_df = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None


if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
            st.session_state.file_type = 'csv'
        else:
            raw_df = pd.read_excel(uploaded_file)
            st.session_state.file_type = 'xlsx'
        
        st.session_state.raw_df = preprocess_raw_data(raw_df)
            
        st.success("✅ File uploaded and read successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head())
        
        # --- Validation Check ---
        fighter_full_name_col = next((c for c in st.session_state.raw_df.columns if ('fighter' in c.lower() or 'player' in c.lower()) and 'full' in c.lower()), None)
        fight_id_col = next((c for c in st.session_state.raw_df.columns if 'fight id' in c.lower() or 'match id' in c.lower()), None)
        date_col = next((c for c in st.session_state.raw_df.columns if 'date' in c.lower()), None)
        
        required_concepts = {
            "Participant Name": fighter_full_name_col,
            "Event ID": fight_id_col,
            "Event Date": date_col,
            "Winner Info": 'Winner' in st.session_state.raw_df.columns,
            "Odds": 'odds' in st.session_state.raw_df.columns
        }

        missing_concepts = [name for name, found in required_concepts.items() if not found]

        if missing_concepts:
            st.error(f"The uploaded file is not compatible. It is missing columns for the following concepts: {', '.join(missing_concepts)}")
        else:
            if st.button("🚀 Generate Training Data"):
                if not all([aggregation_types, last_n_fights_list, outcome_types, betting_statuses]):
                    st.warning("Please make a selection for all feature options.")
                else:
                    with st.spinner(f"Generating features... This may take a moment on the first run."):
                        agg_tuple = tuple(sorted(aggregation_types))
                        last_n_tuple = tuple(sorted(last_n_fights_list))
                        outcome_tuple = tuple(sorted(outcome_types))
                        betting_tuple = tuple(sorted(betting_statuses))
                        st.session_state.training_df = create_training_data(st.session_state.raw_df, agg_tuple, last_n_tuple, shuffle_perspectives, outcome_tuple, betting_tuple)
                    
                    st.success("🎉 Training data generated successfully!")

            if st.session_state.training_df is not None:
                training_df = st.session_state.training_df
                st.subheader("Generated Training Data Preview")
                st.dataframe(training_df)

                file_extension = st.session_state.file_type
                agg_str = '_'.join([agg.lower() for agg in aggregation_types])
                last_n_str = '_'.join(map(str, last_n_fights_list))
                file_name = f'train_data_{agg_str}_last_{last_n_str}.{file_extension}'
                
                if file_extension == 'csv':
                    data = training_df.to_csv(index=False).encode('utf-8')
                    mime = 'text/csv'
                else:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        training_df.to_excel(writer, index=False, sheet_name='Sheet1')
                    data = output.getvalue()
                    mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

                st.download_button(label=f"📥 Download Training Data as .{file_extension}", data=data, file_name=file_name, mime=mime)

                st.subheader("📊 Filtered View")
                min_date, max_date = training_df['date'].min().date(), training_df['date'].max().date()
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)

                select_all_cols = st.checkbox("Select All Columns")
                filtered_cols = training_df.columns.tolist() if select_all_cols else st.multiselect("Select columns to display", training_df.columns.tolist(), default=[])
                
                view_df = training_df
                start_date_ts, end_date_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
                date_mask = (view_df['date'] >= start_date_ts) & (view_df['date'] <= end_date_ts)
                view_df = view_df[date_mask]

                date_filter_active = (start_date != min_date) or (end_date != max_date)
                cols_filter_active = bool(filtered_cols)

                if date_filter_active or cols_filter_active:
                    display_cols = filtered_cols if cols_filter_active else training_df.columns.tolist()
                    st.dataframe(view_df[display_cols])
                else:
                     st.info("Select columns or change the date range to see a filtered view.")
                
                st.subheader("🔍 Fighter Analysis")
                fighters = sorted(training_df['fighter_x_name'].unique())
                col1, col2 = st.columns(2)
                with col1:
                    fighter1 = st.selectbox("Select Fighter 1", fighters, index=None, placeholder="Select fighter...")
                with col2:
                    fighter2 = st.selectbox("Select Fighter 2", fighters, index=None, placeholder="Select fighter...")

                if fighter1 and fighter2 and fighter1 != fighter2:
                    fight_history = training_df[((training_df['fighter_x_name'] == fighter1) & (training_df['fighter_y_name'] == fighter2)) | ((training_df['fighter_x_name'] == fighter2) & (training_df['fighter_y_name'] == fighter1))]
                    if not fight_history.empty:
                        last_fight = fight_history.sort_values('date', ascending=False).iloc[0:1]
                        st.write(f"#### Head-to-Head: Last Matchup")
                        fighter1_perspective = last_fight[last_fight['fighter_x_name'] == fighter1]
                        if fighter1_perspective.empty:
                             fighter1_perspective = last_fight[last_fight['fighter_y_name'] == fighter1].rename(columns=lambda c: c.replace('_y', '_temp').replace('_x', '_y').replace('_temp', '_x'))
                        st.dataframe(fighter1_perspective)
                    else:
                        st.info(f"{fighter1} and {fighter2} have not fought in the selected dataset.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting for a CSV or XLSX file to be uploaded.")
