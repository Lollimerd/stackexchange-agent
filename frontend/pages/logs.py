import streamlit as st
from utils.util import display_container_name, get_system_config

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["🗃Dev Logs🗃", "Contributions by our NSFs", "📈💰📊Dashboard"]
)

with tab1:
    with st.sidebar:
        display_container_name()
        st.sidebar.title("⚙️ Settings", help="config settings here")
        
        # --- ADD THIS EXPANDER FOR SYSTEM DETAILS ---
        with st.expander("System Info & DB Details", expanded=False):
            config_data = get_system_config()
            if config_data:
                st.markdown(
                    f"**Ollama Model:** `{config_data.get('ollama_model', 'N/A')}`"
                )
                st.markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
                st.markdown(
                    f"**DB connected:** `{config_data.get('container_name', 'N/A')}`"
                )
                st.markdown(f"**Neo4j User:** `{config_data.get('neo4j_user', 'N/A')}`")
            else:
                st.error("Could not retrieve system info.")

        st.divider()
        st.markdown("""
        ### Navigation
        [September](#Sep)  
        [August](#Aug)  
        [July](#Jul)
        """)
        st.divider()
        st.write("OPSEC ©LOLLIMERD 2025")

    st.write("=" * 150)
    st.title(
        "**PATCH NOTES / UPDATES / DEV LOGS**",
        help="all dev updates are here",
        anchor="top",
    )
    st.caption("for OSU Personnel only")
    st.write("=" * 150)

    st.header("**`⚒️Admin⚒️ Sep (26/9/25)` 🤖**", anchor="Sep")
    st.subheader("***Phase 2: Louvain Community Detection Algorithm***")
    st.markdown("""
    - ##### Louvain improvements
        - added `tolerance` for adjusting sensitivity of the algorithm, finding more relevant communities
        - attempt to combine `jaccard` with `cosine similarity` as relationship weight for louvain

    - ##### key gds setup 
        - run `apoc` command for node similarity as relationship weights
        ```cypher
        CALL apoc.periodic.iterate(
        // Outer query remains the same
        "MATCH ()-[r]-() RETURN id(r) AS relId",
        
        // Inner query: Now with a robust WHERE clause
        "MATCH (n1)-[r]-(n2) WHERE id(r) = relId 
        AND n1.embeddings IS NOT NULL 
        AND n2.embeddings IS NOT NULL 
        SET r.weight = gds.similarity.cosine(n1.embeddings, n2.embeddings)",
        
        // Configuration remains the same
        {batchSize: 10000, parallel: true}
        )
        YIELD batches, total, timeTaken, committedOperations
        ```
                
        - project gds graph for louvain
        ```cypher
        // Step 1: Project the graph, including the 'weight' property from relationships
        CALL gds.graph.project(
        'community-graph', // Use a new name for the weighted graph
        ['Person', 'Unit', 'Asset', 'Event', 'Exercise', 'Location','Time', 'Title', 'Country', 'Organization', 'Sentence'],
        {
            REL: {
            type: '*',
            orientation: 'NATURAL', // follow native relationsip direction
            // ✨ This tells GDS to load the 'weight' property into memory
            properties: 'weight' 
            }}
        );
        ```
                
        - write louvain algorithm into gds graph
        ```cypher
        CALL gds.louvain.write(
            'community-graph',
            {
                writeProperty: 'communityId',
                relationshipWeightProperty: 'weight', // <-- Corrected to use the 'count' property
                includeIntermediateCommunities: true,
                tolerance: 0.00000000001 // Adjust the sensitivity of the algorithm
            }
        )
        YIELD communityCount, modularity, modularities;
        ```
        
    
    """)
    st.divider()

    st.header("**`⚒️Admin⚒️ Sep (19/9/25)` 🤖**", anchor="Sep")
    st.subheader("***Phase 2: Louvain Community Detection Algorithm***")
    st.markdown("""
    - ##### Hierarchical Louvain clustering
        - added `weighted` relationships for more accurate context via clusters
        - added levels of `hierarchy` due to the knowledge graph `large` nature, able to narrow down context in search of smaller clusters
        - updated repo of saved preset cypher queries collection (will use in each DB instance)
    - ##### successfully debugged problem when break
        - **gitea version control** `master` branch preserved
        - multiple branches for:
            - `tests/experiments` for new testing
            - `archive` to preserve old ver
            - `feature` for adding new features to application
    """)

    st.header("**`⚒️Admin⚒️ Sep (18/9/25)` 🤖**", anchor="Sep")
    st.subheader("***Phase 2: Louvain, Reranking***")
    st.markdown("""
    - ##### adding louvain community detection
        - created `gds graph (in-memory)` for running algorithms
        - use gds graph to write louvain -> add metadata: `communityId` based on cluster
        - plans to add `depths/levels of clustering` during query time
        - heavily accessing via `cypher retrieval query`
    
    - ##### cross encoder reranking model
        - after implement louvain, the number of documents extracted becomes more clustered & increased in numbers
        - model used: `bge-reranker` from ollama & langchain supported tools
        - acts as 2nd step retrieval
    
    - ##### restructured codebase directory to respective components
        - examples are `backend` / `frontend`
        - `utils` is added to not cramp out all functions into backend fastapi file
        - `setup` to store all initialisation steps
        - with addition of `gitea`, repo is set up with more solid config, version control when break
    """)
    st.divider()

    st.header("`⚒️Admin⚒️ Sep (5/9/25)` 🤖", anchor="Sep")
    st.subheader("***Phase 1 continued***")
    st.markdown("""
    - ##### plans of fine-tuning retrieval strategy
        - changed embedding model from `mxbai-embed-large` to `jina/jina-embeddings-v2-base-en:latest`, specialised on RAG
        - configuring different levels of community detection (gds: `louvain`) within cypher query
        - will look into `pagerank` also (GDS ML Algorithm)
        - changed format/feature of context (LLM can read metadata as well)
    """)
    st.divider()

    st.header("`⚒️Admin⚒️ Aug (29/8/25)` 🤖")
    st.subheader("***Agentic + Context Engineering phase 1***")
    st.markdown("""
    - ##### plans of adding `langwatch`, `langsmith` & `langgraph` to the stack for agentic approach
        - found similar sources of ebooks regarding `agentic graphrag` and `fastapi gen ai architecture`

    - ##### Agentic additions features
        - added retrieval router using either `qwen3:8b` or `llama3.1:8b` models, chain of rag and no rag
        - going to implement answer critic soon to the chain using either of the 8b models listed above
        - may change general chat chain with another LLM but now both chains use same LLM `qwen3:30b`
        - plans to add cross encoder for reranking process, either from huggingface or ollama

    - ##### Overall improvements
        - added custom weights to the ensemble retriever
        - attempted to add cross encoder to improve reranking process
        - change `lamba_mult` to **0.5** to strike balance between vector similarity & keyword search
    """)
    st.divider()

    st.header("`⚒️Admin⚒️ Aug (22/8/25)` 🤖")
    st.subheader(
        "***Generating Mermaid Graphs :green-badge[:material/Verified: Mermaid JS]***"
    )
    st.markdown("""
    - ##### Used extension from streamlit library called **`streamlit-mermaid`**
        - generate unique hash key using SHA256 encryption to prevent syntax error when multiple chats of mermaid graphs are present within a chat
        - dedicated function for rendering mermaid code blocks by detecting `some text... ```mermaid`
    - ##### Added dedicated prompt file to expand and finetune LLM for generating Mermaid Graphs (will add more use case in the future)
        - use 4 different libraries: **`SystemMessagePromptTemplate`**, **`HumanMessagePromptTemplate`**, **`AIMessagePromptTemplate`**, **`ChatPromptTemplate`**
        - added a few shot example to test answer generation, will add more examples in the near future
        - improved and expanded to a more robust system prompt, using markdown to emphasize certain critical processes
    """)
    st.divider()

    st.header("`⚒️Admin⚒️ Aug (20/8/25)` 🤖")
    st.subheader("***Finetuning LLM & Embedding model***")
    st.markdown("""
    - ##### added new params for ChatOllama
        - responsible for the answer generation from LLM Qwen3:30b, consistency is key.
            Originally there was a drop in quality of the answers when added `years` to reporting time of dataset
        - added `mirostat` to dynamically control temperature (controlling perplexity)
        - added `num_predict = -2` to prevent over answering, `-2` means to fill context, whatever space the prompt
            use up left, the answer and thoughts takes up the remaining
        - added `repeat_penalty = 1.5` & `repeat_last_n = -1` for using the entire context window for repetition checking with increased penalty
        - used `tfs_z = 2.0` to make sure the model outputs correctly, `2.0` means reduce impact of less probable tokens from output
        - **`top_p = 0.95`** & **`top_k = 100`** work hand in hand to make the model output more diverse answers as everything else is quite narrow down alr
        ```python
        ANSWER_LLM = ChatOllama(
                model="qwen3:30b", # Ensure your model produces <think> tags
                base_url=OLLAMA_BASE_URL,
                num_ctx=262144, # 256k context window
                num_predict=-2, # limit output to fill context window
                tfs_z=2.0, # reduce impact of less probable tokens from output
                repeat_penalty=1.5, # higher, penalise repetitions
                repeat_last_n=-1, # look back within context to penalise penalty
                top_p=0.95, # more diverse text
                top_k=100, # give more diverse answers
                mirostat=2.0, # enable mirostat 2.0 sampling for controlling perplexity
                mirostat_tau=3.0, # output diversity
                mirostat_eta=0.05, # learning rate, responsiveness
        )

        # embedding model
        EMBEDDINGS = OllamaEmbeddings(
                model="mxbai-embed-large",
                base_url=OLLAMA_BASE_URL,
                show_progress=True,
                # tfs_z=2.0, # reduce impact of less probable tokens from output
                mirostat=2.0,
                mirostat_tau=3.0, # output diversity consistent
                mirostat_eta=0.2,
        )
        ```
    - ##### UI updates
        - planning to add anchor for better content navigation
        - thinking on some sort of vector DB for storing chats data, or can do it in neo4j instead
        - adding `buttons` for selecting COY DBs Neo4j

    - ##### Agentic/FastAPI
        - planning to integrate actual langchain agents, instead of stateless (does not remember previous user queries in chat)
        - On temp, running bash script for `streamlit` & `fastapi` instance, run both processes 1 shot
    """)
    st.divider()

    st.header("`⚒️Admin⚒️ Aug (18/8/25)` 🤖", anchor="Aug")
    st.subheader("***Upgrading UI with Streamlit (Official test Launch v1.0.1)***")
    st.markdown("""
    - ##### added new features to the UI
        - added multiple chat feature containing: `add new chats`, `delete` & `clear active history`
        - added system details
        - formatted sidebar config
        - redecorated tags with `st.badge()` like **:red-badge[:material/chat: Test]** using:
            ```python
            st.badge("stackoverflow", icon="🧊") OR :red-badge[:material/component_exchange: Test]
            ```
    - ##### **Plans for agentic approach**
        - retriever router using these 3 tool calls
            - custom tool is up (done) → graph enhanced vector search with custom written retrieval query in cypher
            - gonna generate dynamic graphrag using `GraphCypherQAChain`
            - No RAG, just chat in general
        - plans TBD for answer critic
        - plans TBD for retrieval agent (almost there sorf of)

    - ##### **Gotten help from a professional data assistant**
        - formatting of PS Compile from word doc to csv
        - collating all the years of PS Compile for all COYS
        - implement process of `regex` and `python-docx` for data manipulation
    """)
    st.divider()

    st.header("`⚒️Admin⚒️: July` 🤖", anchor="Jul")
    st.subheader("***Establishing architecture (beta v1.0.0)***")
    st.markdown("""
    - ##### Setting up retrieval query for graph enhanced vector search
        - **NOTEBOOKLM** resources of generating custom cypher queries for graph traversal
            - currently consolidating `videos`, `mindmaps` & `README docs`
        - found several pdfs, ebooks and neo4j courses building this application
    - ##### Configuring api endpoints
        took a while but once succesfully set up on **BCOY**, I replicated it on the other coys
        - NEO4J DB instance (Docker)
            - default: `localhost:7474` & `bolt://localhost:7687`
        - Ollama LLM instance (Docker)
            - default: `localhost:11434` but due to existing applications, it changed to `localhost:11432`
            - changed ports for each coy Neo4j DB
            - plans to dockerise **streamlit frontend** & **fastapi backend**
    - ##### configure scripts for ease of setup
        1. setting up of vectorstores, vector index, langchain libraries and env vars (mainly apis)
        2. choosing ollama embed model `MXBAI-EMBED-LARGE` & answer LLM `Qwen3:30B` with 256k replaced `Qwen3:8b` with 40k context window
        3. Improved neo4j schema of node and relationship declaration, in DB itself and langchain
        4. Optimized, Organised env file, script management in modularity
    """)
    # st.markdown("[Link to my drawing](../info.excalidraw)")
    st.divider()

    st.header("`⚒️Admin⚒️: Jun (Archive)` 🤖", anchor="Jun")
    st.subheader("***Had previous version of tech stack configuration***")
    st.markdown("Its been a while yea...")

    st.divider()
    st.markdown("[Back to the top](#top)")

with tab2:
    st.title("Ops Tech lab team")
    st.subheader("JamesMe0w")
    st.markdown("""
    - ##### Classification Model
    - finetuning llama3.1 instruct 4bit for classification task in progress
    - distillation to RoBERTa code ready for after llama3.1 is finetuned
""")
    st.divider()
    st.subheader("💲💵Kenneth Panglima Broadrick💸💲")
    st.markdown("""
    - ##### DATA COLLECTION
    - Extraction of data from pressum compile
    - Ner model inference
    - Ner model training
    - currently on course in codedex
    """)

    st.subheader("Jason")
    st.markdown("""
    - ##### SUMMARISATION
        - Finished the summarisation inference script for now
        - Fine tuning llama3.1 8B instruct bnb 4bit model 
    - ##### LOCATIONS
        - Gathering locations for SitPic database
        - Script for a heurestic location selector in progress (KIV for now)
        - Nominatim database for higher accuracy (KIV for now)
    - ##### Translator
        - Improving on the fine tuned models (On hold as translator models not deployed yet)
        - Added a UI for demonstration purposes
    - ##### NER Model 
        - Improving on the tokenization/BIO tagging step for the NER model
        - Rewriting CRF-head layer 
        - Trying with a stronger deberta-large-v3 model
    - ##### Entity Pivoting/Pressum view UI
        - Contains Entity pivoting & view for the timeline
        - Completed for now
    """)

with tab3:
    st.write("currently under development, dashboard in progress")
    st.markdown("""""")
    # st_mermaid("""
    # graph TD
    # """)