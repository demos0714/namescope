import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from difflib import get_close_matches
import seaborn as sns

# Set seaborn style for prettier charts
sns.set_style("whitegrid")

# Use absolute path for data folder
DATA_FOLDER = "names"

@st.cache_data(show_spinner=False)
def load_name_data():
    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder '{DATA_FOLDER}' not found. Please ensure the folder exists and contains 'yobXXXX.txt' files.")
        st.stop()
    
    data = []
    for year in range(1880, 2025):
        path = os.path.join(DATA_FOLDER, f"yob{year}.txt")
        try:
            if os.path.exists(path):
                # Try reading with automatic delimiter detection and header handling
                df = pd.read_csv(path, names=["Name", "Gender", "Count"], sep=None, engine='python', header=None)
                # Check if file has a header by inspecting first row
                with open(path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("Name,Gender,Count"):
                        df = pd.read_csv(path, names=["Name", "Gender", "Count"], sep=None, engine='python', skiprows=1)
                # Verify column count
                if len(df.columns) != 3:
                    with open(path, 'r') as f:
                        preview = f.readlines()[:3]
                    st.warning(f"Invalid format in {path}: Expected 3 columns (Name,Gender,Count). Found {len(df.columns)} columns. File preview:\n{''.join(preview)}")
                    continue
                df["Year"] = year
                data.append(df)
        except Exception as e:
            try:
                with open(path, 'r') as f:
                    preview = f.readlines()[:3]
                st.warning(f"Failed to load {path}: {str(e)}. File preview:\n{''.join(preview)}")
            except:
                st.warning(f"Failed to load {path}: {str(e)}. Could not read file for preview.")
            continue
    if not data:
        st.error("No valid data files found in 'names' folder. Ensure files are named 'yobXXXX.txt' with 3 columns: Name,Gender,Count. Example:\nJohn,M,5000\nMary,F,4000")
        st.stop()
    
    df = pd.concat(data)
    df["Name"] = df["Name"].str.lower()
    return df

def describe_lifespan(years):
    if years >= 100: return "a timeless name across generations"
    elif years >= 50: return "a name with decades of popularity"
    else: return "a recently emerging name"

def describe_burst(burst):
    if burst >= 100: return "a massive spike, likely due to cultural or media influence"
    elif burst >= 50: return "a strong surge, driven by trends"
    elif burst >= 20: return "a steady upward trend"
    else: return "slow and organic growth"

def describe_gender_ratio(male, female):
    total = male + female
    ratio = max(male, female) / total if total > 0 else 0
    if ratio >= 0.95: return "strongly tied to one gender"
    elif ratio >= 0.6: return "leans toward one gender, with some crossover"
    else: return "gender-neutral, used by both"

def describe_generation(year_75):
    if year_75 <= 1979: return "common among older generations"
    elif year_75 <= 1999: return "popular with Gen X and Millennials"
    elif year_75 <= 2010: return "a modern name rising recently"
    else: return "a 21st-century trend"

def describe_volatility(score):
    if score >= 0.8: return "highly volatile popularity"
    elif score >= 0.4: return "moderate popularity fluctuations"
    else: return "very stable popularity"

def analyze_name(df_all, name):
    original_name = name
    df_name = df_all[df_all["Name"] == name]
    is_fuzzy = False
    if df_name.empty:
        similar = get_close_matches(name, df_all["Name"].unique(), n=1, cutoff=0.6)
        if similar:
            name = similar[0]
            df_name = df_all[df_all["Name"] == name]
            is_fuzzy = True
    gender_count = df_name.groupby("Gender")["Count"].sum().to_dict()
    male = gender_count.get("M", 0)
    female = gender_count.get("F", 0)
    total = male + female
    df_year = df_name.groupby("Year")["Count"].sum().reset_index()

    first_year = df_year["Year"].min()
    last_year = df_year["Year"].max()
    peak_year = df_year.loc[df_year["Count"].idxmax(), "Year"]
    peak_count = df_year["Count"].max()
    burst = df_year["Count"].pct_change().fillna(0).max()
    volatility = df_year["Count"].std() / df_year["Count"].mean()
    recent_avg = df_year[df_year["Year"] >= 2014]["Count"].mean()
    decline = round((1 - recent_avg / peak_count) * 100, 1) if peak_count else 0

    cum = df_year.sort_values("Year")["Count"].cumsum()
    cutoff = cum.iloc[-1] * 0.75
    year_75 = df_year.iloc[(cum >= cutoff).values.argmax()]["Year"]

    total_counts = df_all.groupby("Name")["Count"].sum().sort_values(ascending=False)
    rank = total_counts.index.get_loc(name) + 1
    rarity = "Very Common" if rank <= 200 else "Moderately Common" if rank <= 2000 else "Rare"
    style = "Retro" if decline > 70 else "Trendy" if peak_year > 2000 else "Classic"

    years = last_year - first_year + 1
    summary = {
        "First Appearance": f"{first_year}",
        "Last Recorded": f"{last_year}",
        "Timespan": f"{years} years - {describe_lifespan(years)}",
        "Peak Year": f"{peak_year} with {peak_count:,} births",
        "Max Growth": f"{round(burst*100,1)}% - {describe_burst(burst*100)}",
        "Gender Distribution": f"Boys: {male:,}, Girls: {female:,} - {describe_gender_ratio(male, female)}",
        "Main Era": f"75% usage before {year_75} - {describe_generation(year_75)}",
        "Popularity Stability": f"Score {round(volatility,2)} - {describe_volatility(volatility)}",
        "Rank": f"#{rank:,} of {len(total_counts):,} - {rarity}",
        "Style": style
    }

    return {
        "name": name,
        "original_name": original_name,
        "is_fuzzy": is_fuzzy,
        "gender": "Male" if male >= female else "Female",
        "confidence": round(max(male, female) / total * 100, 1) if total > 0 else 50,
        "total": total,
        "male": male,
        "female": female,
        "first_year": first_year,
        "last_year": last_year,
        "peak_year": peak_year,
        "peak_count": peak_count,
        "burst": round(burst * 100, 1),
        "volatility": round(volatility, 2),
        "decline": decline,
        "year_75": year_75,
        "rank": rank,
        "rarity": rarity,
        "style": style,
        "df_year": df_year,
        "narrative": summary
    }

# Streamlit app (top-level code, no indentation)
st.set_page_config(page_title="NameScope", layout="wide")
st.title("🔍 NameScope")
st.caption("Discover the era, trends, and gender signals behind names")

all_data = load_name_data()

input_names = st.text_input("Enter one or more English names (comma-separated):")

if input_names:
    names = [n.strip().lower() for n in input_names.split(',') if n.strip()]
    if len(names) > 5:
        st.warning("⚠️ For optimal visualization, please enter up to 5 names.")
        names = names[:5]
    analyses = [analyze_name(all_data, name) for name in names]

    # Dropdown menu for selecting name
    selected_name = st.selectbox("Select a name to analyze:", 
                                [a["name"].capitalize() for a in analyses])
    
    # Display analysis for selected name
    selected_analysis = next(a for a in analyses if a["name"].capitalize() == selected_name)
    
    st.subheader(f"🧠 Analysis of {selected_name}")
    
    # Fuzzy match warning
    if selected_analysis["is_fuzzy"]:
        st.warning(f"⚠️ '{selected_analysis['original_name'].capitalize()}' not found. "
                   f"Showing results for '{selected_name}' instead.")

    # Display analysis in a table
    st.markdown("### Name Insights")
    narrative = selected_analysis["narrative"]
    summary_df = pd.DataFrame({
        "Metric": narrative.keys(),
        "Details": narrative.values()
    })
    st.table(summary_df)

    # Charts (side-by-side)
    st.subheader("📊 Trend Charts")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        for a in analyses:
            ax1.plot(a['df_year']['Year'], a['df_year']['Count'], 
                    label=a['name'].capitalize(), alpha=0.3 if a['name'] != selected_analysis['name'] else 1)
        ax1.set_title("Trend Over Time")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Births")
        ax1.grid(True)
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=False)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        # Prepare data for stacked area chart
        years = range(2014, 2025)
        data = {a['name'].capitalize(): [0] * len(years) for a in analyses}
        for a in analyses:
            recent = a['df_year'][a['df_year']['Year'] >= 2014]
            for _, row in recent.iterrows():
                if row['Year'] in years:
                    data[a['name'].capitalize()][years.index(row['Year'])] = row['Count']
        df_area = pd.DataFrame(data, index=years)
        ax2.stackplot(years, df_area.T, labels=df_area.columns, 
                     alpha=0.8, baseline='zero')
        ax2.set_title("Usage in Last 10 Years")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Births")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=False)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        df_ranks = all_data.groupby(["Year", "Name"])["Count"].sum().reset_index()
        df_ranks["Rank"] = df_ranks.groupby("Year")["Count"].rank(ascending=False)
        for a in analyses:
            name_ranks = df_ranks[df_ranks["Name"] == a['name']]
            ax3.plot(name_ranks["Year"], name_ranks["Rank"], 
                    label=a['name'].capitalize(), alpha=0.3 if a['name'] != selected_analysis['name'] else 1)
        ax3.invert_yaxis()
        ax3.set_title("Popularity Rank")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Rank")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=False)

else:
    st.info("💡 Please enter one or more names (comma-separated) to start analysis.")
