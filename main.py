import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

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

def calculate_implied_probability(odds):
    """Calculates the implied win probability from American odds."""
    if odds >= 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_historical_stats(fighter_df, aggregation_types, last_n_fights):
    """
    Calculates historical stats for a single fighter's past performances,
    including breakdowns for when they were the favorite or underdog.
    """
    # Use only the last N fights if specified. If 0, it uses the whole dataframe.
    if last_n_fights > 0:
        fighter_df = fighter_df.tail(last_n_fights)

    # --- Basic Career Stats ---
    stats = {}
    wins = fighter_df[fighter_df['Winner']]
    losses = fighter_df[~fighter_df['Winner']]
    stats['NumberOf_Fight'] = len(fighter_df)
    stats['NumberOf_WIN'] = len(wins)
    stats['NumberOf_LOSE'] = len(losses)
    stats['WIN_RATE'] = stats['NumberOf_WIN'] / stats['NumberOf_Fight'] if stats['NumberOf_Fight'] > 0 else 0

    # --- NEW: Segment fights by favorite/underdog status ---
    favorite_fights = fighter_df[fighter_df['odds'] < 0]
    underdog_fights = fighter_df[fighter_df['odds'] >= 0]

    # Further segment by win/loss
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
        agg_func_name = agg_type.lower()[:3] if agg_type == 'Average' else agg_type.lower()
        agg_method = agg_map[agg_type]
        
        for col in stat_columns:
            for suffix, (win_df, lose_df) in segments.items():
                win_stat_name = f'win_{agg_func_name}_{col.replace(" ", "_")}{suffix}'
                lose_stat_name = f'lose_{agg_func_name}_{col.replace(" ", "_")}{suffix}'
                
                stats[win_stat_name] = win_df[col].agg(agg_method) if not win_df.empty else 0
                stats[lose_stat_name] = lose_df[col].agg(agg_method) if not lose_df.empty else 0

    return pd.Series(stats).fillna(0)


def create_training_data(df, aggregation_types, last_n_fights):
    """
    Transforms the raw fight data into a training-ready format with organized columns.
    This logic creates TWO rows per fight, one for each fighter's perspective.
    """
    processed_fights = []
    progress_bar = st.progress(0)
    total_rows = len(df)

    # Iterate through every row in the raw data. Each row is one fighter in a fight.
    for i, fighter_x_info in df.iterrows():
        fight_id = fighter_x_info['Fight ID']
        fight_date = fighter_x_info['date']
        
        # Find the opponent's row using a more robust index-based method
        opponent_query = df[(df['Fight ID'] == fight_id) & (df.index != i)]
        
        if opponent_query.empty:
            continue
        fighter_y_info = opponent_query.iloc[0]

        # --- Get historical data for each fighter ---
        fighter_x_history = df[(df['Fighter Full Name'] == fighter_x_info['Fighter Full Name']) & (df['date'] < fight_date)]
        fighter_y_history = df[(df['Fighter Full Name'] == fighter_y_info['Fighter Full Name']) & (df['date'] < fight_date)]
        
        # --- Calculate historical stats ---
        fighter_x_stats = calculate_historical_stats(fighter_x_history, aggregation_types, last_n_fights)
        fighter_y_stats = calculate_historical_stats(fighter_y_history, aggregation_types, last_n_fights)
        
        # --- Calculate Static Features for each fighter ---
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

        # --- Robust odds and probability calculation ---
        odds_x = fighter_x_info.get('odds', 0)
        odds_y = fighter_y_info.get('odds', 0)
        prob_x = calculate_implied_probability(odds_x)
        prob_y = calculate_implied_probability(odds_y)
        diff_odds = odds_x - odds_y

        # --- Calculate differential stats ---
        diff_stats = (fighter_x_stats - fighter_y_stats).add_prefix('diff_')
        diff_static = {f"diff_{k.replace('_x','')}": v - static_features_y.get(k.replace('_x','_y'), 0) for k,v in static_features_x.items() if k != 'Stance_x'}
        
        # --- Add suffixes to historical stats ---
        fighter_x_stats = fighter_x_stats.add_suffix('_x')
        fighter_y_stats = fighter_y_stats.add_suffix('_y')
        
        # --- Organize data into logical blocks ---
        fight_info = pd.Series({
            'fight_id': fight_id, 
            'date': fight_date, 
            'fighter_x_name': fighter_x_info['Fighter Full Name'],
            'fighter_y_name': fighter_y_info['Fighter Full Name'], 
            'fighter_x_win': int(fighter_x_info['Winner'])
        })
        fighter_x_data = pd.concat([pd.Series(static_features_x), fighter_x_stats, pd.Series({'odds_x': odds_x, 'implied_probability_x': prob_x})])
        fighter_y_data = pd.concat([pd.Series(static_features_y), fighter_y_stats, pd.Series({'odds_y': odds_y, 'implied_probability_y': prob_y})])
        diff_data = pd.concat([pd.Series(diff_static), diff_stats, pd.Series({'diff_odds': diff_odds})])

        combined_stats = pd.concat([fight_info, fighter_x_data, fighter_y_data, diff_data])
        processed_fights.append(combined_stats)

        progress_bar.progress((i + 1) / total_rows)

    final_df = pd.DataFrame(processed_fights).fillna(0)
    
    # Sort the final dataframe by fight_id to ensure sequential and consistent order
    final_df = final_df.sort_values(by=['fight_id', 'fighter_x_name']).reset_index(drop=True)
    
    return final_df


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title('🥊 UFC Fight Data Feature Engineering')

st.markdown("""
This app transforms raw, fight-by-fight UFC data into a "training-ready" dataset with advanced filtering and analysis. 
**How it works:**
1.  **Upload** your `raw.csv` or `raw.xlsx` file.
2.  **Choose** your feature engineering parameters on the left. You can now select multiple aggregation types.
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
    
    last_n_fights = st.number_input(
        "Number of Last Fights to Consider (0 for all)",
        min_value=0,
        value=10,
        step=1,
        help="Calculate stats based on the last N fights. Set to 0 to use a fighter's entire history."
    )
    st.info("ℹ️ To match the original `train.csv`, select only 'Average' and set 'Last Fights' to 0.")

    # --- NEW: Date Filter ---
    st.header("📅 Date Filter")
    if 'raw_df' in st.session_state and st.session_state.raw_df is not None:
        min_date = st.session_state.raw_df['date'].min().date()
        max_date = st.session_state.raw_df['date'].max().date()
        date_range = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    else:
        st.info("Upload a file to enable date filtering.")
        date_range = None


# --- Main App Logic ---
if 'training_df' not in st.session_state:
    st.session_state.training_df = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None


if uploaded_file is not None:
    try:
        # Read the uploaded file based on its type
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
            st.session_state.file_type = 'csv'
        else:
            raw_df = pd.read_excel(uploaded_file)
            st.session_state.file_type = 'xlsx'
        
        # Preprocess once and store in session state
        st.session_state.raw_df = preprocess_raw_data(raw_df.copy())
            
        st.success("✅ File uploaded and read successfully!")

        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head())

        if st.button("🚀 Generate Training Data"):
            if not aggregation_types:
                st.warning("Please select at least one aggregation type.")
            else:
                # --- Apply Date Filter ---
                filtered_df = st.session_state.raw_df
                if date_range and len(date_range) == 2:
                    start_date = pd.to_datetime(date_range[0])
                    end_date = pd.to_datetime(date_range[1])
                    filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

                with st.spinner(f"Generating features... This is a long process, please wait."):
                    st.session_state.training_df = create_training_data(filtered_df, aggregation_types, last_n_fights)
                
                st.success("🎉 Training data generated successfully!")

        if st.session_state.training_df is not None:
            training_df = st.session_state.training_df
            st.subheader("Generated Training Data Preview")

            # --- NEW: Column Selector with "ALL" option ---
            show_all = st.checkbox("Show All Columns", value=False)
            
            if show_all:
                visible_cols = training_df.columns.tolist()
                st.multiselect("Select columns to display", training_df.columns.tolist(), default=visible_cols, disabled=True)
            else:
                default_cols = [col for col in training_df.columns if not ('favorite' in col or 'underdog' in col)]
                visible_cols = st.multiselect("Select columns to display", training_df.columns.tolist(), default=default_cols)
            
            if visible_cols:
                st.dataframe(training_df[visible_cols])
            else:
                st.info("Select one or more columns to display the data.")

            
            # --- Dynamic Download Button ---
            file_extension = st.session_state.file_type
            agg_str = '_'.join([agg.lower() for agg in aggregation_types])
            file_name = f'train_data_{agg_str}_last_{last_n_fights}.{file_extension}'
            
            if file_extension == 'csv':
                data = training_df.to_csv(index=False).encode('utf-8')
                mime = 'text/csv'
            else: # xlsx
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

            # --- Fighter Analysis Section ---
            st.subheader("🔍 Fighter Analysis")
            fighters = sorted(training_df['fighter_x_name'].unique())
            
            col1, col2 = st.columns(2)
            with col1:
                fighter1 = st.selectbox("Select Fighter 1", fighters, index=0)
            with col2:
                fighter2 = st.selectbox("Select Fighter 2", fighters, index=1)

            if fighter1 and fighter2:
                # Find the last fight between them
                fight_history = training_df[
                    ((training_df['fighter_x_name'] == fighter1) & (training_df['fighter_y_name'] == fighter2)) |
                    ((training_df['fighter_x_name'] == fighter2) & (training_df['fighter_y_name'] == fighter1))
                ]

                if not fight_history.empty:
                    last_fight = fight_history.sort_values('date', ascending=False).iloc[0:1]
                    st.write(f"#### Head-to-Head: Last Matchup")
                    
                    # Ensure fighter1 is always fighter_x for consistent display
                    fighter1_perspective = last_fight[last_fight['fighter_x_name'] == fighter1]
                    if fighter1_perspective.empty:
                         st.info(f"No direct matchup found where {fighter1} is listed as fighter_x. Showing opponent's perspective.")
                         fighter1_perspective = last_fight[last_fight['fighter_y_name'] == fighter1]


                    st.dataframe(fighter1_perspective[visible_cols])
                else:
                    st.info(f"{fighter1} and {fighter2} have not fought in the selected dataset.")


    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting for a CSV or XLSX file to be uploaded.")
