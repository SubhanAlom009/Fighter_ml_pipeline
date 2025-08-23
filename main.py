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
    df['date'] = pd.to_datetime(df['date'])
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
    
    # Expanded list of all columns that should be numeric for calculations.
    numeric_cols = [
        'Winning Time', 'Height Feet', 'Height Inches', 'Weight Pounds', 
        'Reach Inches', 'odds', 'Knockdown Total', 
        'Significant Strike Total Attempted', 'Significant Strike Total Landed',
        'Takedown Total Attempted', 'Takedown Total Landed', 'Submission Attempted', 
        'Reversal', 'Ground and Cage Control Time', 'Significant Strike Head Attempted', 
        'Significant Strike Head Landed', 'Significant Strike Body Attempted', 
        'Significant Strike Body Landed', 'Significant Strike Leg Attempted', 
        'Significant Strike Leg Landed', 'Significant Strike Clinch Attempted', 
        'Significant Strike Clinch Landed', 'Significant Strike Ground Attempted', 
        'Significant Strike Ground Landed'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Clean up Stance - fill missing values
    df['Stance'] = df['Stance'].fillna('Unknown')

    # Combine winner's first and last name to get a full name for comparison
    df['Winner Full Name'] = (df['Winner First Name'].astype(str) + ' ' + df['Winner Last Name'].astype(str)).str.upper()
    df['Winner'] = df['Fighter Full Name'].str.upper() == df['Winner Full Name']
    
    # Sort by date AND Fight ID to create a stable, guaranteed chronological order
    df = df.sort_values(by=['date', 'Fight ID']).reset_index(drop=True)
    return df

def calculate_historical_stats(fighter_df, aggregation_types, last_n_fights_list):
    """
    Calculates historical stats for a single fighter's past performances for multiple lookback windows.
    """
    all_stats = {}
    
    for last_n in last_n_fights_list:
        # Use only the last N fights if specified. If 0, it uses the whole dataframe.
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
        win_favorite = favorite_fights[favorite_fights['Winner']]
        lose_favorite = favorite_fights[~favorite_fights['Winner']]
        win_underdog = underdog_fights[underdog_fights['Winner']]
        lose_underdog = underdog_fights[~underdog_fights['Winner']]

        # --- Columns to aggregate ---
        stat_columns = [
            'Winning Time', 'Knockdown Total', 'Significant Strike Total Attempted', 'Significant Strike Total Landed',
            'Takedown Total Attempted', 'Takedown Total Landed', 'Submission Attempted', 'Reversal',
            'Ground and Cage Control Time', 'Significant Strike Head Attempted', 'Significant Strike Head Landed',
            'Significant Strike Body Attempted', 'Significant Strike Body Landed', 'Significant Strike Leg Attempted',
            'Significant Strike Leg Landed', 'Significant Strike Clinch Attempted', 'Significant Strike Clinch Landed',
            'Significant Strike Ground Attempted', 'Significant Strike Ground Landed'
        ]

        # --- Aggregation Logic ---
        agg_map = {'Average': 'mean', 'Sum': 'sum', 'Median': 'median'}

        # --- Calculate stats for all segments ---
        segments = {
            '': (wins, losses),
            '_favorite': (win_favorite, lose_favorite),
            '_underdog': (win_underdog, lose_underdog)
        }

        for agg_type in aggregation_types:
            agg_func_name = 'avg' if agg_type == 'Average' else agg_type.lower()
            agg_method = agg_map[agg_type]
            
            for col in stat_columns:
                for suffix, (win_df, lose_df) in segments.items():
                    win_stat_name = f'win_{agg_func_name}_{col.replace(" ", "_")}{suffix}_{last_n}'
                    lose_stat_name = f'lose_{agg_func_name}_{col.replace(" ", "_")}{suffix}_{last_n}'
                    
                    all_stats[win_stat_name] = win_df[col].agg(agg_method) if not win_df.empty else 0
                    all_stats[lose_stat_name] = lose_df[col].agg(agg_method) if not lose_df.empty else 0

    return pd.Series(all_stats).fillna(0)


# --- Caching Decorators for Optimization ---
@st.cache_data
def create_training_data(_df, aggregation_types, last_n_fights_list, shuffle_perspectives):
    """
    Transforms the raw fight data into a training-ready format with organized columns.
    This logic creates TWO rows per fight, one for each fighter's perspective.
    """
    df = _df.copy()
    processed_fights = []
    
    progress_bar_placeholder = st.sidebar.empty()
    progress_bar_placeholder.progress(0)
    total_rows = len(df)

    # Iterate through every row in the raw data.
    for i, fighter_x_info in df.iterrows():
        fight_id = fighter_x_info['Fight ID']
        fight_date = fighter_x_info['date']
        
        opponent_query = df[(df['Fight ID'] == fight_id) & (df.index != i)]
        if opponent_query.empty:
            continue
        fighter_y_info = opponent_query.iloc[0]

        fighter_x_history = df[(df['Fighter Full Name'] == fighter_x_info['Fighter Full Name']) & (df['date'] < fight_date)]
        fighter_y_history = df[(df['Fighter Full Name'] == fighter_y_info['Fighter Full Name']) & (df['date'] < fight_date)]
        
        fighter_x_stats = calculate_historical_stats(fighter_x_history, aggregation_types, last_n_fights_list)
        fighter_y_stats = calculate_historical_stats(fighter_y_history, aggregation_types, last_n_fights_list)
        
        static_features_x = {
            'Fighter ID_x': fight_id,
            'Age (in days)_x': (fight_date - fighter_x_info['Date of Birth']).days if pd.notna(fighter_x_info['Date of Birth']) else 0,
            'Height_x': fighter_x_info['Height Feet'] * 12 + fighter_x_info['Height Inches'],
            'Weight Pounds_x': fighter_x_info['Weight Pounds'],
            'Reach Inches_x': fighter_x_info['Reach Inches'],
            'Stance_x': fighter_x_info['Stance'],
            'DOB Month_x': fighter_x_info['Date of Birth'].month if pd.notna(fighter_x_info['Date of Birth']) else 0,
            'DOB Day_x': fighter_x_info['Date of Birth'].day if pd.notna(fighter_x_info['Date of Birth']) else 0,
            'DOB Year_x': fighter_x_info['Date of Birth'].year if pd.notna(fighter_x_info['Date of Birth']) else 0,
        }
        static_features_y = {
            'Fighter ID_y': fight_id,
            'Age (in days)_y': (fight_date - fighter_y_info['Date of Birth']).days if pd.notna(fighter_y_info['Date of Birth']) else 0,
            'Height_y': fighter_y_info['Height Feet'] * 12 + fighter_y_info['Height Inches'],
            'Weight Pounds_y': fighter_y_info['Weight Pounds'],
            'Reach Inches_y': fighter_y_info['Reach Inches'],
            'Stance_y': fighter_y_info['Stance'],
            'DOB Month_y': fighter_y_info['Date of Birth'].month if pd.notna(fighter_y_info['Date of Birth']) else 0,
            'DOB Day_y': fighter_y_info['Date of Birth'].day if pd.notna(fighter_y_info['Date of Birth']) else 0,
            'DOB Year_y': fighter_y_info['Date of Birth'].year if pd.notna(fighter_y_info['Date of Birth']) else 0,
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
            'fight_id': fight_id, 'date': fight_date, 'fighter_x_name': fighter_x_info['Fighter Full Name'],
            'fighter_y_name': fighter_y_info['Fighter Full Name'], 'fighter_x_win': int(fighter_x_info['Winner'])
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
        for _, group in final_df.groupby('fight_id'):
            shuffled_list.append(group.sample(frac=1))
        final_df = pd.concat(shuffled_list).reset_index(drop=True)
    else:
        final_df = final_df.sort_values(by=['fight_id', 'fighter_x_name']).reset_index(drop=True)
    
    return final_df


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title('🥊 UFC Fight Data Feature Engineering')

st.markdown("""
This app transforms raw, fight-by-fight UFC data into a "training-ready" dataset with advanced filtering and analysis. 
**How it works:**
1.  **Upload** your `raw.csv` or `raw.xlsx` file.
2.  **Choose** your feature engineering parameters on the left. You can now select multiple aggregation and lookback types.
3.  The app calculates historical stats, odds, and differentials for each fighter *before* every fight.
4.  Use the new analysis and filtering tools to explore the generated data.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Feature Engineering Controls")
    
    uploaded_file = st.file_uploader("Upload Raw Data File", type=["csv", "xlsx"])
    
    aggregation_types = st.multiselect(
        "Select Aggregation Types",
        ['Average', 'Sum', 'Median'],
        default=['Average'],
        help="Choose one or more ways to summarize stats."
    )
    
    last_n_fights_list = st.multiselect(
        "Select 'Last N Fights' Windows",
        [0, 3, 5, 10, 20],
        default=[10],
        help="Select one or more lookback periods. 0 means entire career history."
    )

    shuffle_perspectives = st.checkbox("Shuffle fighter perspectives (x vs y)", value=False, help="Randomize the order of the two rows for each fight.")

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

        if st.button("🚀 Generate Training Data"):
            if not aggregation_types:
                st.warning("Please select at least one aggregation type.")
            elif not last_n_fights_list:
                st.warning("Please select at least one 'Last N Fights' window.")
            else:
                with st.spinner(f"Generating features... This may take a moment on the first run."):
                    agg_tuple = tuple(sorted(aggregation_types))
                    last_n_tuple = tuple(sorted(last_n_fights_list))
                    st.session_state.training_df = create_training_data(st.session_state.raw_df, agg_tuple, last_n_tuple, shuffle_perspectives)
                
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

            st.download_button(
                label=f"📥 Download Training Data as .{file_extension}",
                data=data,
                file_name=file_name,
                mime=mime,
            )

            st.subheader("📊 Filtered View")
            
            min_date = training_df['date'].min().date()
            max_date = training_df['date'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)

            select_all_cols = st.checkbox("Select All Columns")
            
            if select_all_cols:
                filtered_cols = training_df.columns.tolist()
            else:
                filtered_cols = st.multiselect(
                    "Select columns to display", 
                    training_df.columns.tolist(), 
                    default=[]
                )
            
            view_df = training_df
            
            start_date_ts = pd.to_datetime(start_date)
            end_date_ts = pd.to_datetime(end_date)
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
                fight_history = training_df[
                    ((training_df['fighter_x_name'] == fighter1) & (training_df['fighter_y_name'] == fighter2)) |
                    ((training_df['fighter_x_name'] == fighter2) & (training_df['fighter_y_name'] == fighter1))
                ]

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
