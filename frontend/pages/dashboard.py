import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from utils.util import get_database_summary, get_import_history, display_container_name

# Load environment variables
load_dotenv()

# Neo4j connection
url = os.getenv("NEO4J_URL")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")

neo4j_graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password,
)

def render_page():
    st.header("📊 StackOverflow Import Dashboard")
    st.caption("Track your StackOverflow data imports and database statistics")
    
    # Display container status in sidebar
    with st.sidebar:
        display_container_name()
    
    # Get database summary
    try:
        summary = get_database_summary(neo4j_graph)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📝 Total Questions",
                value=f"{summary['total_questions']:,}",
                help="Total questions imported from StackOverflow"
            )
        
        with col2:
            st.metric(
                label="🏷️ Total Tags", 
                value=f"{summary['total_tags']:,}",
                help="Unique tags in the database"
            )
        
        with col3:
            st.metric(
                label="💬 Total Answers",
                value=f"{summary['total_answers']:,}",
                help="Total answers imported"
            )
        
        with col4:
            st.metric(
                label="👥 Total Users",
                value=f"{summary['total_users']:,}",
                help="Unique users in the database"
            )
        
        # Additional metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📦 Import Sessions",
                value=f"{summary['total_imports']:,}",
                help="Total import sessions recorded"
            )
        
        with col2:
            last_import = summary.get('last_import')
            if last_import:
                # Format the datetime for display
                if hasattr(last_import, 'strftime'):
                    formatted_date = last_import.strftime("%Y-%m-%d %H:%M")
                else:
                    formatted_date = str(last_import)[:16]  # Truncate if it's a string
                st.metric(
                    label="🕒 Last Import",
                    value=formatted_date,
                    help="Date of the most recent import session"
                )
            else:
                st.metric(
                    label="🕒 Last Import",
                    value="Never",
                    help="No import sessions recorded yet"
                )
        
        with col3:
            # Calculate average questions per import
            if summary['total_imports'] > 0:
                avg_questions = summary['total_questions'] / summary['total_imports']
                st.metric(
                    label="📈 Avg Questions/Import",
                    value=f"{avg_questions:.1f}",
                    help="Average questions imported per session"
                )
            else:
                st.metric(
                    label="📈 Avg Questions/Import",
                    value="N/A",
                    help="No import sessions yet"
                )
        
    except Exception as e:
        st.error(f"Could not fetch database summary: {e}")
        return
    
    st.divider()
    
    # Import History Section
    st.subheader("📋 Import History")
    
    try:
        # Get import history
        history = get_import_history(neo4j_graph, limit=20)
        
        if history:
            # Convert to DataFrame for better display
            df = pd.DataFrame(history)
            
            # Format timestamp for display
            if 'timestamp' in df.columns:
                df['formatted_time'] = df['timestamp'].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, 'strftime') else str(x)[:16]
                )
            
            # Display as table
            st.dataframe(
                df[['formatted_time', 'questions', 'tags', 'pages', 'tags_list']].rename(columns={
                    'formatted_time': 'Date',
                    'questions': 'Questions',
                    'tags': 'Tags',
                    'pages': 'Pages',
                    'tags_list': 'Tag List'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Create visualization
            if len(df) > 1:
                st.subheader("📈 Import Trends")
                
                # Questions imported over time
                fig = px.bar(
                    df, 
                    x='formatted_time', 
                    y='questions',
                    title="Questions Imported Over Time",
                    labels={'formatted_time': 'Import Date', 'questions': 'Questions Imported'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tags distribution
                if len(df) > 0:
                    # Count unique tags across all imports
                    all_tags = []
                    for tags_list in df['tags_list']:
                        if tags_list:
                            all_tags.extend(tags_list)
                    
                    if all_tags:
                        tag_counts = pd.Series(all_tags).value_counts().head(10)
                        fig2 = px.bar(
                            x=tag_counts.values,
                            y=tag_counts.index,
                            orientation='h',
                            title="Most Imported Tags (Top 10)",
                            labels={'x': 'Import Count', 'y': 'Tag'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No import history found. Start importing data to see statistics here.")
            
    except Exception as e:
        st.error(f"Could not fetch import history: {e}")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📥 Go to Loader", use_container_width=True):
            st.switch_page("pages/loader.py")

render_page()
