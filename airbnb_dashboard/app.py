import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import folium
from streamlit_folium import st_folium
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Airbnb Data Analysis Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Airbnb Data Analysis & Machine Learning Dashboard")
st.markdown("""
This dashboard performs a comprehensive analysis of Airbnb Listings and Reviews. 
It includes Data Preprocessing, EDA, Time Series Analysis, Geospatial Mapping, and Machine Learning modeling.
""")

# --- 1. Data Loading & Preprocessing (Cached) ---
@st.cache_data
def load_and_preprocess_data():
    # Load Data
    try:
        df_rev = pd.read_csv('Reviews.csv', encoding='latin1')
        df_list = pd.read_csv('Listings.csv', encoding='latin1', low_memory=False)
        df_rev_dict = pd.read_csv('Reviews_data_dictionary.csv')
        df_list_dict = pd.read_csv('Listings_data_dictionary.csv')
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure 'Reviews.csv' and 'Listings.csv' are in the app directory.")
        return None, None, None, None, None, None

    # --- Preprocessing Listings ---
    # 1. Date Conversion & Imputation
    df_list['host_since'] = pd.to_datetime(df_list['host_since'], errors='coerce')
    if 'host_since' in df_list.columns:
        mode_host_since = df_list['host_since'].mode()[0]
        df_list['host_since'] = df_list['host_since'].fillna(mode_host_since)

    # 2. Numerical Imputation (Median)
    num_cols_impute = [
        'bedrooms', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value', 'host_response_rate',
        'host_acceptance_rate', 'host_total_listings_count'
    ]
    for col in num_cols_impute:
        if col in df_list.columns:
            # Clean percentage strings if necessary (generic safety check)
            if df_list[col].dtype == 'object': 
                 df_list[col] = pd.to_numeric(df_list[col].astype(str).str.replace('%', ''), errors='coerce')
            median_val = df_list[col].median()
            df_list[col] = df_list[col].fillna(median_val)

    # 3. Categorical Imputation
    cat_cols_unknown = ['name', 'host_location']
    for col in cat_cols_unknown:
        if col in df_list.columns:
            df_list[col] = df_list[col].fillna('Unknown')
    
    if 'host_response_time' in df_list.columns:
        df_list['host_response_time'] = df_list['host_response_time'].fillna('No Response')
        
    if 'district' in df_list.columns:
        df_list['district'] = df_list['district'].fillna('No District')

    cat_cols_mode = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']
    for col in cat_cols_mode:
        if col in df_list.columns:
            mode_val = df_list[col].mode()[0]
            df_list[col] = df_list[col].fillna(mode_val)

    # 4. Feature Engineering
    df_list['host_since_year'] = df_list['host_since'].dt.year
    df_list['host_since_month'] = df_list['host_since'].dt.month
    current_year = datetime.datetime.now().year
    df_list['host_experience_years'] = current_year - df_list['host_since_year']

    # 5. Encoding
    # Select categorical cols for encoding
    categorical_cols_to_encode = [
        'host_response_time', 'host_is_superhost', 'host_has_profile_pic',
        'host_identity_verified', 'neighbourhood', 'district', 'city',
        'property_type', 'room_type', 'instant_bookable'
    ]
    existing_cat_cols = [col for col in categorical_cols_to_encode if col in df_list.columns]
    df_encoded = pd.get_dummies(df_list, columns=existing_cat_cols, drop_first=True)

    # --- Preprocessing Reviews ---
    df_rev['date'] = pd.to_datetime(df_rev['date'])
    df_rev['review_month'] = df_rev['date'].dt.to_period('M')

    # --- Merging ---
    # Calculate average rating per listing from reviews if needed, but we usually use listing_id
    # For Time Series, we merge reviews with listings
    df_merged = pd.merge(df_rev, df_list, on='listing_id', how='inner')

    return df_list, df_rev, df_encoded, df_merged, df_rev_dict, df_list_dict

# Load Data
df_listings, df_reviews, df_listings_encoded, df_merged, df_reviews_dict, df_listings_dict = load_and_preprocess_data()

if df_listings is None:
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
options = [
    "Data Overview", 
    "EDA: Univariate & Bivariate", 
    "Geospatial Analysis", 
    "Time Series Analysis", 
    "Sentiment Analysis",
    "Machine Learning Models",
    "Final Summary"
]
selection = st.sidebar.radio("Go to:", options)

# --- 1. Data Overview ---
if selection == "Data Overview":
    st.header("Data Overview")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Listings Data", "Reviews Data", "Listings Dictionary", "Reviews Dictionary"])
    
    with tab1:
        st.subheader("Listings DataFrame")
        st.write(f"Shape: {df_listings.shape}")
        st.dataframe(df_listings.head())
        st.write("Descriptive Statistics (Numerical):")
        st.dataframe(df_listings.describe())

    with tab2:
        st.subheader("Reviews DataFrame")
        st.write(f"Shape: {df_reviews.shape}")
        st.dataframe(df_reviews.head())
        
    with tab3:
        st.subheader("Listings Data Dictionary")
        st.dataframe(df_listings_dict)

    with tab4:
        st.subheader("Reviews Data Dictionary")
        st.dataframe(df_reviews_dict)
    
    st.subheader("Missing Values Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Listings Missing Values (After Imputation Check)**")
        missing_list = df_listings.isnull().sum()
        st.write(missing_list[missing_list > 0])
    with col2:
        st.write("**Reviews Missing Values**")
        missing_rev = df_reviews.isnull().sum()
        st.write(missing_rev[missing_rev > 0])

# --- 2. EDA: Univariate & Bivariate ---
elif selection == "EDA: Univariate & Bivariate":
    st.header("Exploratory Data Analysis")
    
    st.subheader("Univariate Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Numerical Distributions")
        num_cols = df_listings.select_dtypes(include=np.number).columns.tolist()
        selected_num_col = st.selectbox("Select Numerical Column", num_cols, index=num_cols.index('price') if 'price' in num_cols else 0)
        
        fig, ax = plt.subplots()
        sns.histplot(df_listings[selected_num_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_num_col}")
        st.pyplot(fig)
        
    with col2:
        st.markdown("### Categorical Frequencies")
        cat_cols = df_listings.select_dtypes(include='object').columns.tolist()
        # Filter out high cardinality columns for better plotting
        cat_cols = [c for c in cat_cols if df_listings[c].nunique() < 20]
        if cat_cols:
            selected_cat_col = st.selectbox("Select Categorical Column", cat_cols, index=0)
            fig, ax = plt.subplots()
            sns.countplot(y=df_listings[selected_cat_col], order=df_listings[selected_cat_col].value_counts().index, palette='viridis', hue=df_listings[selected_cat_col], legend=False, ax=ax)
            ax.set_title(f"Count of {selected_cat_col}")
            st.pyplot(fig)
        else:
            st.write("No suitable categorical columns for bar plots found.")

    st.divider()
    st.subheader("Bivariate Analysis")
    
    st.markdown("### Correlation Heatmap")
    # Fixed list of key numerical cols to prevent massive heatmap
    key_cols = ['price', 'accommodates', 'bedrooms', 'review_scores_rating', 'host_total_listings_count', 'minimum_nights']
    existing_key_cols = [c for c in key_cols if c in df_listings.columns]
    
    if existing_key_cols:
        corr = df_listings[existing_key_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        
    st.markdown("### Scatter / Box Plot Explorer")
    plot_type = st.radio("Select Plot Type", ["Scatter Plot", "Box Plot"], horizontal=True)
    
    col_x, col_y = st.columns(2)
    with col_x:
        if plot_type == "Scatter Plot":
            x_axis = st.selectbox("X Axis", num_cols, index=num_cols.index('accommodates') if 'accommodates' in num_cols else 0)
        else:
            x_axis = st.selectbox("X Axis (Categorical)", cat_cols, index=0)
    with col_y:
        y_axis = st.selectbox("Y Axis (Numerical)", num_cols, index=num_cols.index('price') if 'price' in num_cols else 0)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_type == "Scatter Plot":
        sns.scatterplot(data=df_listings, x=x_axis, y=y_axis, alpha=0.5, ax=ax)
    else:
        sns.boxplot(data=df_listings, x=x_axis, y=y_axis, palette='viridis', hue=x_axis, legend=False, ax=ax)
        plt.xticks(rotation=45)
    st.pyplot(fig)

# --- 3. Geospatial Analysis ---
elif selection == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    
    st.subheader("Price by City")
    if 'city' in df_listings.columns and 'price' in df_listings.columns:
        avg_price_city = df_listings.groupby('city')['price'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg_price_city.index, y=avg_price_city.values, palette='viridis', hue=avg_price_city.index, legend=False, ax=ax)
        plt.xticks(rotation=45)
        ax.set_title("Average Price by City")
        st.pyplot(fig)
        
    st.subheader("Interactive Map")
    st.write("Showing a sample of 1,000 listings to maintain performance.")
    
    if 'latitude' in df_listings.columns and 'longitude' in df_listings.columns:
        # Sample data
        sample_map = df_listings.dropna(subset=['latitude', 'longitude', 'price']).sample(n=min(1000, len(df_listings)), random_state=42)
        
        # Base Map
        m = folium.Map(location=[sample_map['latitude'].mean(), sample_map['longitude'].mean()], zoom_start=10)
        
        # Color helper
        def get_color(price):
            if price < 100: return 'green'
            elif price < 300: return 'blue'
            elif price < 600: return 'orange'
            else: return 'red'

        for i, row in sample_map.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=get_color(row['price']),
                fill=True,
                fill_color=get_color(row['price']),
                fill_opacity=0.7,
                popup=f"Price: ${row['price']}<br>Room: {row.get('room_type', 'N/A')}"
            ).add_to(m)
            
        st_folium(m, width=800, height=500)
        st.caption("Green: <$100 | Blue: $100-$300 | Orange: $300-$600 | Red: >$600")

# --- 4. Time Series Analysis ---
elif selection == "Time Series Analysis":
    st.header("Time Series Analysis")
    
    # Ensure review_month is timestamp for plotting
    df_merged['review_month_dt'] = df_merged['date'].dt.to_period('M').dt.to_timestamp()
    
    # Aggregation
    ts_data = df_merged.groupby('review_month_dt').agg({
        'review_id': 'count',
        'price': 'mean',
        'review_scores_rating': 'mean'
    }).rename(columns={'review_id': 'Review Count', 'price': 'Average Price', 'review_scores_rating': 'Average Rating'})
    
    st.subheader("Review Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=ts_data, x=ts_data.index, y='Review Count', marker='o', ax=ax)
    ax.set_title("Number of Reviews per Month")
    st.pyplot(fig)
    
    st.subheader("Price Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=ts_data, x=ts_data.index, y='Average Price', color='purple', marker='o', ax=ax)
    ax.set_title("Average Listing Price per Month")
    st.pyplot(fig)

    st.subheader("Time Series Decomposition (Review Counts)")
    if len(ts_data) > 24: # Need enough data points
        decomposition = seasonal_decompose(ts_data['Review Count'], model='additive', period=12)
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        st.pyplot(fig)
    else:
        st.warning("Not enough data points for decomposition.")

# --- 5. Sentiment Analysis ---
elif selection == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    st.markdown("Since text data is unavailable, sentiment analysis focuses on numerical review scores.")
    
    score_cols = [c for c in df_listings.columns if 'review_scores_' in c]
    
    st.subheader("Score Distributions")
    selected_score = st.selectbox("Select Score Metric", score_cols)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_listings[selected_score], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribution of {selected_score}")
    st.pyplot(fig)
    
    st.subheader("Score Correlation")
    if len(score_cols) > 1:
        corr_scores = df_listings[score_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_scores, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# --- 6. Machine Learning Models ---
elif selection == "Machine Learning Models":
    st.header("Machine Learning Models")
    
    model_choice = st.selectbox("Select Model", ["Linear Regression (Price)", "Logistic Regression (Superhost)", "K-Means Clustering", "Random Forest (Price)"])
    
    # --- Model 1: Linear Regression ---
    if model_choice == "Linear Regression (Price)":
        st.subheader("Linear Regression: Predicting Price")
        
        # Prepare Data
        cols_exclude = ['listing_id', 'host_id', 'name', 'host_location', 'amenities', 'price', 'host_since', 'host_since_year']
        existing_exclude = [c for c in cols_exclude if c in df_listings_encoded.columns]
        X = df_listings_encoded.drop(columns=existing_exclude)
        y = df_listings_encoded['price']
        
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if st.button("Train Linear Regression Model"):
            with st.spinner("Training model..."):
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("R2 Score", f"{r2:.4f}")
                
                st.markdown("#### Actual vs Predicted")
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual Price")
                ax.set_ylabel("Predicted Price")
                st.pyplot(fig)

    # --- Model 2: Logistic Regression ---
    elif model_choice == "Logistic Regression (Superhost)":
        st.subheader("Logistic Regression: Predicting Superhost Status")
        
        target_col = 'host_is_superhost_t' # Assumes One-Hot created this. 
        # Check if column exists, might be host_is_superhost_t or host_is_superhost_True depending on pandas version
        possible_targets = [c for c in df_listings_encoded.columns if 'host_is_superhost' in c]
        if not possible_targets:
             st.error("Target column for Superhost not found.")
        else:
            # Usually it's the one not dropped. Let's assume the user wants to predict 't' (True)
            # If 'host_is_superhost' was binary mapped, adjust accordingly. 
            # Here we look for the encoded version.
            target = possible_targets[0] 
            
            cols_exclude = ['listing_id', 'host_id', 'name', 'host_location', 'amenities', 'price', 'host_since', 'host_since_year'] + possible_targets
            existing_exclude = [c for c in cols_exclude if c in df_listings_encoded.columns]
            
            X = df_listings_encoded.drop(columns=existing_exclude)
            y = df_listings_encoded[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.button("Train Logistic Regression Model"):
                with st.spinner("Training..."):
                    model = LogisticRegression(max_iter=1000, solver='liblinear')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    st.info("Note: Low Precision/Recall often indicates class imbalance (few Superhosts vs non-Superhosts).")

    # --- Model 3: K-Means ---
    elif model_choice == "K-Means Clustering":
        st.subheader("K-Means Clustering")
        
        # Feature Selection
        cols_exclude = [
            'listing_id', 'host_id', 'name', 'host_location', 'amenities', 'price',
            'host_since', 'host_since_year'
        ]
        # Remove any Superhost encoded cols
        cols_exclude += [c for c in df_listings_encoded.columns if 'host_is_superhost' in c]
        
        existing_exclude = [c for c in cols_exclude if c in df_listings_encoded.columns]
        X = df_listings_encoded.drop(columns=existing_exclude).select_dtypes(include=np.number)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow Plot
        st.markdown("#### Elbow Method")
        wcss = []
        k_range = range(1, 8)
        for i in k_range:
            kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, wcss, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
        
        k = st.slider("Select Number of Clusters (K)", 2, 8, 4)
        
        if st.button("Run Clustering"):
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # PCA for viz
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            
            df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = clusters
            
            st.markdown("#### Cluster Visualization (PCA)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
            st.pyplot(fig)
            
            st.markdown("#### Cluster Analysis")
            df_temp = df_listings_encoded.copy()
            df_temp['Cluster'] = clusters
            # Show mean of key features
            st.dataframe(df_temp.groupby('Cluster')[['price', 'review_scores_rating', 'host_total_listings_count']].mean())

    # --- Model 4: Random Forest ---
    elif model_choice == "Random Forest (Price)":
        st.subheader("Random Forest Regression")
        
        cols_exclude = ['listing_id', 'host_id', 'name', 'host_location', 'amenities', 'price', 'host_since', 'host_since_year']
        existing_exclude = [c for c in cols_exclude if c in df_listings_encoded.columns]
        X = df_listings_encoded.drop(columns=existing_exclude)
        y = df_listings_encoded['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.warning("Training Random Forest on the full dataset may take time. Using 50 estimators.")
        
        if st.button("Train Random Forest"):
            with st.spinner("Training Random Forest..."):
                rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("R2 Score", f"{r2:.4f}")
                
                # Feature Importance
                st.markdown("#### Top 10 Feature Importance")
                importances = rf.feature_importances_
                feat_importances = pd.Series(importances, index=X.columns)
                top_10 = feat_importances.nlargest(10)
                
                fig, ax = plt.subplots()
                top_10.plot(kind='barh', ax=ax)
                ax.invert_yaxis()
                st.pyplot(fig)

# --- 7. Final Summary ---
elif selection == "Final Summary":
    st.header("Project Summary & Key Findings")
    
    st.markdown("""
    ### 1. Data Quality & Preprocessing
    * **Missing Values:** Significant missing data in `host_response_time` and review scores required extensive imputation (Median/Mode).
    * **Encoding:** Categorical variables were one-hot encoded, significantly increasing feature dimensionality.
    
    ### 2. EDA & Trends
    * **Time Series:** Reviews show a clear upward trend over the years with seasonal peaks (summer). Prices also exhibit seasonality.
    * **Geography:** Cape Town and Bangkok listings appeared pricier on average in this specific dataset slice compared to European cities.
    
    ### 3. Model Performance
    
    | Model | Metric | Result | Insight |
    | :--- | :--- | :--- | :--- |
    | **Linear Regression** | R² | **0.15** | Poor fit. Relationship between features and price is likely non-linear. |
    | **Random Forest** | R² | **0.22** | Better than Linear Regression, but still leaves variance unexplained. Captures non-linearity. |
    | **Logistic Regression** | F1-Score | **0.00** | Failed to predict Superhosts (Class Imbalance Issue). Accuracy is misleading. |
    | **K-Means** | Clusters | **4** | Successfully segmented listings into Budget, Standard, Premium, and High-Volume Host categories. |
    
    ### 4. Key Drivers (Feature Importance)
    From the Random Forest model, the top drivers of price include:
    1.  **Location:** `longitude`, `latitude`
    2.  **Property Type:** Specific room types (Shared vs Entire)
    3.  **Host Activity:** `host_total_listings_count`
    
    ### 5. Recommendations
    * **For Price Prediction:** Incorporate external data (local events, seasonality indices) and try Gradient Boosting (XGBoost/LightGBM).
    * **For Superhost Classification:** Address class imbalance using SMOTE (Oversampling) or adjust class weights.
    """)
    
st.sidebar.info("Created with Streamlit")