import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from utils.util import get_database_summary, get_import_history, display_container_name, update_import_log_api, delete_import_log_api

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
    st.header("📊 StackExchange Import Dashboard")
    st.caption("Track your StackExchange data imports and database statistics")
    
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
                help="Total questions imported from StackExchange"
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
            
            # Display as interactive editor
            edited_df = st.data_editor(
                df[['id', 'formatted_time', 'questions', 'tags', 'pages', 'tags_list']].rename(columns={
                    'formatted_time': 'Date',
                    'questions': 'Questions',
                    'tags': 'Tags',
                    'pages': 'Pages',
                    'tags_list': 'Tag List'
                }),
                key="import_history_editor",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": None,  # Hide ID column
                    "Date": st.column_config.DatetimeColumn(
                        "Date",
                        disabled=True,
                    ),
                    "Tag List": st.column_config.ListColumn(
                        "Tag List",
                        help="List of tags imported",
                    )
                },
                disabled=["Date", "Tags"],  # Disable editing of date and calculated tags count
                num_rows="dynamic"  # Allow adding/deleting rows (we only handle deletion)
            )

            # Handle changes
            if st.session_state.get("import_history_editor"):
                changes = st.session_state["import_history_editor"]
                
                # Handle updates
                for index, updates in changes.get("edited_rows", {}).items():
                    # Get the ID from the original dataframe 
                    # Note: index is the row index in the displayed dataframe
                    row_id = df.iloc[index]['id']
                    
                    # Synthesize new data
                    current_row = df.iloc[index].to_dict()
                    
                    # Map column names back to internal names for updates
                    col_map_inv = {
                        'Questions': 'total_questions',
                        'Tags': 'total_tags', 
                        'Pages': 'total_pages',
                        'Tag List': 'tags_list'
                    }
                    
                    update_data = {}
                    for col, val in updates.items():
                        if col in col_map_inv:
                            update_data[col_map_inv[col]] = val
                    
                    # We might need other fields if the backend expects a full object, 
                    # but our backend endpoint accepts partial updates via params matching ImportRecordRequest structure?
                    # Wait, ImportRecordRequest is:
                    # class ImportRecordRequest(BaseModel):
                    #     total_questions: int
                    #     tags_list: List[str]
                    #     total_pages: int
                    
                    # The backend update endpoint expects a full ImportRecordRequest object.
                    # So we need to reconstruct the full object from existing + updates.
                    
                    full_payload = {
                        "total_questions": update_data.get("total_questions", current_row.get("questions")),
                        "tags_list": update_data.get("tags_list", current_row.get("tags_list")),
                        "total_pages": update_data.get("total_pages", current_row.get("pages"))
                    }

                    if update_import_log_api(row_id, full_payload):
                        st.success(f"Updated row {index + 1}")
                        st.rerun()

                # Handle deletions
                # st.data_editor with num_rows="fixed" prevents deletion via UI (delete key). 
                # To support deletion we need num_rows="dynamic" to allow deletions, 
                # BUT that also allows adding rows which we said we didn't want essentially?
                # Actually, `num_rows="dynamic"` allows adding and deleting.
                # If we want ONLY delete, that's tricky with standard data_editor.
                # A common pattern is adding a "Delete" checkbox column or similar.
                # Or just allowing dynamic and ignoring additions (or validating them).
                
                # Let's try to stick to the plan: "allow user to update, modify or delete".
                # If I use num_rows="dynamic", user can delete.
                
                for index in changes.get("deleted_rows", []):
                     row_id = df.iloc[index]['id']
                     if delete_import_log_api(row_id):
                         st.success(f"Deleted row {index + 1}")
                         st.rerun()

            
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
