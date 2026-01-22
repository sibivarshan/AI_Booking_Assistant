"""
AI Booking Assistant - Admin Dashboard
View and manage all bookings with filtering capabilities.
"""

import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Admin Dashboard - AI Booking Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path
DB_PATH = Path(__file__).parent.parent / "booking.db"

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-confirmed {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .status-pending {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-cancelled {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)


def get_all_bookings():
    """Fetch all bookings from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
            SELECT 
                b.id as 'Booking ID',
                c.name as 'Customer Name',
                c.email as 'Email',
                c.phone as 'Phone',
                b.booking_type as 'Booking Type',
                b.date as 'Date',
                b.time as 'Time',
                b.status as 'Status',
                b.created_at as 'Created At'
            FROM bookings b
            JOIN customers c ON b.customer_id = c.customer_id
            ORDER BY b.created_at DESC
            """
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        st.error(f"Error fetching bookings: {str(e)}")
        return pd.DataFrame()


def get_booking_stats():
    """Get booking statistics."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Total bookings
            cursor.execute("SELECT COUNT(*) FROM bookings")
            total = cursor.fetchone()[0]
            
            # Today's bookings - compare as string since date is stored as TEXT
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT COUNT(*) FROM bookings WHERE date LIKE ?", (today + '%',))
            today_count = cursor.fetchone()[0]
            
            # Confirmed bookings
            cursor.execute("SELECT COUNT(*) FROM bookings WHERE status = 'confirmed'")
            confirmed = cursor.fetchone()[0]
            
            # Pending bookings
            cursor.execute("SELECT COUNT(*) FROM bookings WHERE status = 'pending'")
            pending = cursor.fetchone()[0]
            
            # Total customers
            cursor.execute("SELECT COUNT(*) FROM customers")
            customers = cursor.fetchone()[0]
            
            return {
                "total": total,
                "today": today_count,
                "confirmed": confirmed,
                "pending": pending,
                "customers": customers
            }
    except Exception as e:
        return {
            "total": 0,
            "today": 0,
            "confirmed": 0,
            "pending": 0,
            "customers": 0
        }


def update_booking_status(booking_id: int, new_status: str):
    """Update the status of a booking."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE bookings SET status = ? WHERE id = ?",
                (new_status, booking_id)
            )
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error updating booking: {str(e)}")
        return False


def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.page_link("streamlit_app.py", label="💬 Chat", icon="💬")
        st.page_link("pages/admin.py", label="📊 Admin Dashboard", icon="📊")
        
        st.markdown("---")
        
        st.markdown("## 📊 Dashboard Info")
        st.markdown("""
        Manage and view all bookings:
        - 📋 View all bookings
        - 🔍 Filter by name, email, date
        - ✏️ Update booking status
        - 📈 View statistics
        """)


def render_metrics(stats):
    """Render the metrics cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📅 Total Bookings",
            value=stats["total"],
            delta=None
        )
    
    with col2:
        st.metric(
            label="📆 Today",
            value=stats["today"],
            delta=None
        )
    
    with col3:
        st.metric(
            label="✅ Confirmed",
            value=stats["confirmed"],
            delta=None
        )
    
    with col4:
        st.metric(
            label="⏳ Pending",
            value=stats["pending"],
            delta=None
        )
    
    with col5:
        st.metric(
            label="👥 Customers",
            value=stats["customers"],
            delta=None
        )


def render_filters():
    """Render the filter controls."""
    st.markdown("### 🔍 Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_name = st.text_input("Search by Name", placeholder="Enter name...")
    
    with col2:
        search_email = st.text_input("Search by Email", placeholder="Enter email...")
    
    with col3:
        date_from = st.date_input(
            "Date From",
            value=datetime.now() - timedelta(days=30),
            key="date_from"
        )
    
    with col4:
        date_to = st.date_input(
            "Date To",
            value=datetime.now() + timedelta(days=30),
            key="date_to"
        )
    
    status_filter = st.multiselect(
        "Filter by Status",
        options=["confirmed", "pending", "cancelled"],
        default=["confirmed", "pending"]
    )
    
    return {
        "name": search_name,
        "email": search_email,
        "date_from": date_from,
        "date_to": date_to,
        "status": status_filter
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filters to the bookings dataframe."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Filter by name
    if filters["name"]:
        filtered_df = filtered_df[
            filtered_df["Customer Name"].str.lower().str.contains(filters["name"].lower(), na=False)
        ]
    
    # Filter by email
    if filters["email"]:
        filtered_df = filtered_df[
            filtered_df["Email"].str.lower().str.contains(filters["email"].lower(), na=False)
        ]
    
    # Filter by date range
    if "Date" in filtered_df.columns:
        # Convert to datetime for comparison
        try:
            # Handle various date formats
            filtered_df["Date_parsed"] = pd.to_datetime(filtered_df["Date"], errors='coerce')
            date_from = pd.to_datetime(filters["date_from"])
            date_to = pd.to_datetime(filters["date_to"])
            
            # Only filter rows with valid dates
            valid_dates = filtered_df["Date_parsed"].notna()
            filtered_df = filtered_df[
                ~valid_dates | ((filtered_df["Date_parsed"] >= date_from) & (filtered_df["Date_parsed"] <= date_to))
            ]
            # Drop the temporary column
            filtered_df = filtered_df.drop(columns=["Date_parsed"])
        except Exception as e:
            # If date parsing fails, don't filter by date
            if "Date_parsed" in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=["Date_parsed"])
            pass
    
    # Filter by status
    if filters["status"]:
        filtered_df = filtered_df[
            filtered_df["Status"].str.lower().isin([s.lower() for s in filters["status"]])
        ]
    
    return filtered_df


def render_bookings_table(df: pd.DataFrame):
    """Render the bookings table with actions."""
    if df.empty:
        st.info("📭 No bookings found matching your filters.")
        return
    
    st.markdown(f"### 📋 Bookings ({len(df)} records)")
    
    # Display the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Booking ID": st.column_config.NumberColumn("ID", width="small"),
            "Customer Name": st.column_config.TextColumn("Customer", width="medium"),
            "Email": st.column_config.TextColumn("Email", width="medium"),
            "Phone": st.column_config.TextColumn("Phone", width="medium"),
            "Booking Type": st.column_config.TextColumn("Type", width="medium"),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Created At": st.column_config.TextColumn("Created", width="medium"),
        }
    )
    
    # Status update section
    st.markdown("---")
    st.markdown("### ✏️ Update Booking Status")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        booking_ids = df["Booking ID"].tolist()
        if booking_ids:
            selected_id = st.selectbox("Select Booking ID", booking_ids)
        else:
            selected_id = None
    
    with col2:
        new_status = st.selectbox(
            "New Status",
            options=["confirmed", "pending", "cancelled"]
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Update Status", type="primary", disabled=not selected_id):
            if update_booking_status(selected_id, new_status):
                st.success(f"✅ Booking #{selected_id} status updated to '{new_status}'")
                st.rerun()


def render_export_section(df: pd.DataFrame):
    """Render the export section."""
    if df.empty:
        return
    
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export to CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"bookings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export to JSON
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_str,
            file_name=f"bookings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main admin dashboard entry point."""
    render_sidebar()
    
    # Header
    st.markdown("# 📊 Admin Dashboard")
    st.markdown("Manage and monitor all bookings in one place.")
    
    st.markdown("---")
    
    # Metrics
    stats = get_booking_stats()
    render_metrics(stats)
    
    st.markdown("---")
    
    # Filters
    filters = render_filters()
    
    st.markdown("---")
    
    # Get and filter bookings
    df = get_all_bookings()
    filtered_df = apply_filters(df, filters)
    
    # Bookings table
    render_bookings_table(filtered_df)
    
    # Export section
    render_export_section(filtered_df)


if __name__ == "__main__":
    main()
