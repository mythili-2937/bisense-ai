import streamlit as st
import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from ai_analyzer import analyze_requirement, explain_recommendations
from pdf_analyzer import extract_text_from_pdf
from report_generator import generate_pdf_report


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="BISense AI",
    page_icon="🧠",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --navy: #0F172A;
        --navy-2: #172554;
        --indigo: #4338CA;
        --blue: #2563EB;
        --teal: #0F766E;
        --bg: #EEF2F7;
        --card: #FFFFFF;
        --text: #1E293B;
        --muted: #64748B;
        --border: #D9E2EC;
        --success: #15803D;
        --warning: #B45309;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(67,56,202,0.08), transparent 22%),
            radial-gradient(circle at 88% 12%, rgba(37,99,235,0.08), transparent 20%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #172554 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 0.42rem 0.35rem;
        border-radius: 10px;
        transition: 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: var(--navy);
        margin-bottom: 2px;
        letter-spacing: -0.8px;
    }

    .subtitle {
        font-size: 18px;
        color: var(--muted);
        margin-top: 0;
        margin-bottom: 24px;
    }

    .hero-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 52%, #4338CA 100%);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 24px;
        padding: 34px;
        margin-bottom: 28px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        color: #FFFFFF;
    }

    .hero-box .main-title {
        color: #FFFFFF;
        font-size: 48px;
    }

    .hero-box .subtitle {
        color: #DBEAFE;
        margin-bottom: 10px;
        font-size: 20px;
    }

    .hero-box p {
        color: #E2E8F0;
        font-size: 15px;
        margin-bottom: 0;
    }

    .hero-chip {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 7px;
        margin-top: 14px;
    }

    .metric-card {
        background: rgba(255,255,255,0.98);
        border: 1px solid #DCE5EF;
        border-radius: 18px;
        padding: 20px 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.07);
        text-align: center;
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.05;
    }

    .metric-label {
        font-size: 13px;
        color: var(--muted);
        margin-top: 8px;
        font-weight: 600;
    }

    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.055);
    }

    .feature-card {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 18px;
        padding: 20px;
        min-height: 150px;
        box-shadow: 0 8px 22px rgba(15,23,42,0.05);
    }

    .feature-title {
        font-weight: 800;
        color: var(--navy);
        font-size: 17px;
        margin-bottom: 6px;
    }

    .feature-text {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.55;
    }

    .status-good {
        color: var(--success);
        font-weight: 700;
    }

    .status-warning {
        color: var(--warning);
        font-weight: 700;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
        border: 1px solid transparent;
    }

    .badge-blue {
        background: #DBEAFE;
        color: #1D4ED8;
        border-color: #BFDBFE;
    }

    .badge-teal {
        background: #CCFBF1;
        color: #0F766E;
        border-color: #99F6E4;
    }

    .badge-amber {
        background: #FEF3C7;
        color: #B45309;
        border-color: #FDE68A;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #2563EB, #4338CA);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        min-height: 44px;
        box-shadow: 0 8px 18px rgba(37,99,235,0.18);
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(37,99,235,0.24);
        color: white;
        border: none;
    }

    textarea, input {
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #94A3B8;
        border-radius: 16px;
        padding: 10px;
    }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(15,23,42,0.05);
    }

    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 12px;
        padding-top: 34px;
        padding-bottom: 10px;
    }

    .sidebar-status {
        margin-top: 16px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 12px;
        line-height: 1.8;
    }

    h1, h2, h3 {
        color: var(--navy);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# PATHS
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "standards.csv"


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv(CSV_PATH)

df["search_text"] = (
    df["title"].fillna("") + ". " +
    df["product"].fillna("") + ". " +
    df["category"].fillna("") + ". " +
    df["scope"].fillna("") + ". " +
    df["keywords"].fillna("")
)


# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -------------------------------------------------
# FAISS INDEX
# -------------------------------------------------

@st.cache_resource
def build_index(search_text):

    embeddings = model.encode(
        search_text,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


index = build_index(df["search_text"].tolist())


# -------------------------------------------------
# SEARCH FUNCTION
# -------------------------------------------------

def clean_ai_output(text):
    """Remove stray HTML tags from LLM output while keeping Markdown readable."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def search_standards(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if score < 0.30:
            continue

        row = df.iloc[idx]

        results.append({
            "standard_id": row["standard_id"],
            "title": row["title"],
            "product": row["product"],
            "category": row["category"],
            "scope": row["scope"],
            "score": round(float(score) * 100, 2),
            "related_standards": (
                "" if pd.isna(row["related_standards"])
                else row["related_standards"]
            ),
            "certification": (
                "" if pd.isna(row["certification"])
                else row["certification"]
            )
        })

    return results


def get_related_standards(results):
    related_results = []
    seen_ids = set()

    for result in results:
        related_ids = result.get("related_standards", "")

        if not related_ids:
            continue

        ids = [x.strip() for x in str(related_ids).split(";") if x.strip()]

        for related_id in ids:
            if related_id in seen_ids:
                continue

            match = df[
                df["standard_id"].astype(str).str.strip() == related_id
            ]

            if match.empty:
                continue

            row = match.iloc[0]

            related_results.append({
                "standard_id": row["standard_id"],
                "title": row["title"],
                "product": row["product"],
                "category": row["category"],
                "scope": row["scope"],
                "certification": (
                    ""
                    if pd.isna(row["certification"])
                    else row["certification"]
                )
            })

            seen_ids.add(related_id)

    return related_results


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.markdown("## 🧠 BISense AI")

st.sidebar.caption(
    "AI-Powered Standards Recommendation Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Analyze Requirement",
        "Tender Analyzer",
        "Standards Explorer",
        "Compliance Summary",
        "About"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-status">
        <b>Prototype Status</b><br>
        ● AI Engine Active<br>
        ● PDF Analyzer Ready<br>
        ● Multilingual Ready<br>
        ● PDF Reports Ready
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("Internal Hackathon Prototype")


# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero-box">
            <div class="main-title">BISense AI</div>
            <div class="subtitle">AI-Powered Standards Intelligence for Procurement</div>
            <p>
                Understand requirements, recommend standards, discover allied references,
                and generate explainable compliance insights from text or tender documents.
            </p>
            <span class="hero-chip">Semantic AI</span>
            <span class="hero-chip">Tender Analysis</span>
            <span class="hero-chip">Multilingual</span>
            <span class="hero-chip">PDF Reports</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{len(df)}</div>
                <div class="metric-label">Prototype Standards</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{df['category'].nunique()}</div>
                <div class="metric-label">Categories</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">AI</div>
                <div class="metric-label">Semantic Analysis</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">Instant</div>
                <div class="metric-label">Recommendations</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Core Capabilities")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Semantic Recommendation</div>
                <div class="feature-text">
                    Understands procurement intent and retrieves relevant standards beyond exact keywords.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Tender Analyzer</div>
                <div class="feature-text">
                    Upload a PDF tender and automatically extract product, specifications, and compliance needs.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Allied Standards</div>
                <div class="feature-text">
                    Resolves related safety, testing, and supporting standards from the knowledge base.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Compliance Report</div>
                <div class="feature-text">
                    Summarizes recommendations and generates a downloadable PDF for review.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## How It Works")

    st.markdown(
        """
        **1. Enter Requirement**  
        Product specification, procurement need, or technical description.

        **2. AI Understanding**  
        Extract product, category, specifications, and application.

        **3. Semantic Search**  
        Match the requirement with relevant standards.

        **4. Compliance Intelligence**  
        Show related standards and certification information.

        **5. Explainable Recommendation**  
        AI explains why each standard is relevant.
        """
    )


# -------------------------------------------------
# ANALYZE REQUIREMENT PAGE
# -------------------------------------------------

elif page == "Analyze Requirement":

    st.markdown(
        '<div class="main-title">Analyze Requirement</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Describe a procurement need and let BISense AI interpret, retrieve, and explain relevant standards.</div>',
        unsafe_allow_html=True
    )

    language = st.selectbox(
        "Response Language",
        ["English", "Tamil", "Hindi"],
        key="manual_response_language"
    )

    query = st.text_area(
        "Procurement Requirement",
        height=160,
        placeholder=(
            "Example: Need 90W outdoor LED street lights "
            "with IP66 protection for highway use..."
        )
    )

    analyze_button = st.button(
        "Analyze with AI",
        use_container_width=True
    )

    if analyze_button:

        if not query.strip():

            st.warning("Please enter a procurement requirement.")

        else:

            with st.spinner("AI is understanding the requirement..."):

                analysis = analyze_requirement(query, language)

            st.success("Requirement analyzed successfully.")

            st.markdown("## AI Understanding")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class="section-card">
                    <b>Product</b><br>
                    {analysis["product"]}
                    <br><br>

                    <b>Category</b><br>
                    {analysis["category"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                specs = ", ".join(
                    analysis["specifications"]
                )

                st.markdown(
                    f"""
                    <div class="section-card">
                    <b>Application</b><br>
                    {analysis["application"]}
                    <br><br>

                    <b>Specifications</b><br>
                    {specs}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            compliance = ", ".join(
                analysis["compliance_needs"]
            )

            st.markdown(
                f"""
                <div class="section-card">
                <b>Compliance Needs</b><br>
                {compliance if compliance else "Not explicitly identified"}
                </div>
                """,
                unsafe_allow_html=True
            )

            search_query = analysis["search_query"]

            results = search_standards(search_query)
            related_results = get_related_standards(results)

            st.session_state["latest_analysis"] = {
                "source": "Manual Requirement",
                "query": query,
                "analysis": analysis,
                "results": results,
                "related_results": related_results,
                "language": language
            }

            st.markdown("## Recommended Standards")
            st.markdown(
                '<span class="badge badge-blue">AI Retrieved</span>'
                '<span class="badge badge-teal">Semantic Match</span>',
                unsafe_allow_html=True
            )

            if not results:

                st.warning(
                    "No strong matching standards found in the prototype dataset."
                )

            else:

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    with st.container():

                        st.markdown(
                            f"### {i}. {result['standard_id']} — {result['title']}"
                        )

                        st.markdown(
                            f"""
**Product:** {result['product']}

**Category:** {result['category']}

**Semantic Match:** {result['score']}%

**Certification:** {result['certification'] or 'Not available'}
"""
                        )

                        with st.expander(
                            "View more details"
                        ):

                            st.write(
                                "**Scope:**",
                                result["scope"]
                            )

                            st.write(
                                "**Related Standards:**",
                                (
                                    result["related_standards"]
                                    or "Not available"
                                )
                            )

                st.markdown("## Allied Standards")

                if not related_results:
                    st.info(
                        "No additional allied standards were found "
                        "in the prototype knowledge base."
                    )
                else:
                    for i, standard in enumerate(related_results, start=1):
                        st.markdown(
                            f"### {i}. {standard['standard_id']} — {standard['title']}"
                        )

                        st.markdown(
                            f"""
**Product:** {standard['product']}

**Category:** {standard['category']}

**Certification:** {standard['certification'] or 'Not available'}
"""
                        )

                        with st.expander(
                            f"View scope for {standard['standard_id']}"
                        ):
                            st.write(standard["scope"])

                with st.spinner(
                    "Generating AI explanation..."
                ):

                    explanation = explain_recommendations(
                        query,
                        results,
                        language
                    )

                explanation = clean_ai_output(explanation)

                st.session_state["latest_analysis"]["explanation"] = explanation

                st.markdown(
                    "## AI Recommendation Explanation"
                )

                st.markdown(
                    """
                    <div class="section-card">
                        <b>AI Explanation</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Render the AI response as normal Markdown so HTML-like tags
                # are not exposed in the UI.
                st.markdown(explanation)

                st.info(
                    "Prototype results are based on a demo standards knowledge base. "
                    "Final recommendations should be verified against official BIS sources."
                )

elif page == "Tender Analyzer":

    st.markdown(
        '<div class="main-title">Tender Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Upload a tender document and automatically extract requirements, standards, and compliance context.</div>',
        unsafe_allow_html=True
    )

    language = st.selectbox(
        "Response Language",
        ["English", "Tamil", "Hindi"],
        key="tender_response_language"
    )

    uploaded_file = st.file_uploader(
        "Upload Tender Document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button(
            "Analyze Tender with AI",
            use_container_width=True
        ):

            # ---------------------------------
            # Extract tender text
            # ---------------------------------
            with st.spinner("Extracting text from tender..."):
                tender_text = extract_text_from_pdf(uploaded_file)

            if not tender_text:

                st.error(
                    "No readable text was found in this PDF. "
                    "If this is a scanned PDF, OCR support will be needed."
                )

            else:

                st.success("Tender text extracted successfully.")

                # ---------------------------------
                # Tender Statistics
                # ---------------------------------
                word_count = len(tender_text.split())
                character_count = len(tender_text)

                col1, col2 = st.columns(2)

                col1.metric(
                    "Words Extracted",
                    word_count
                )

                col2.metric(
                    "Characters Extracted",
                    character_count
                )

                # ---------------------------------
                # Preview extracted content
                # ---------------------------------
                with st.expander("View Extracted Tender Text"):

                    st.text_area(
                        "Tender Content",
                        value=tender_text[:8000],
                        height=300,
                        disabled=True
                    )

                # ---------------------------------
                # Limit text sent to Gemini
                # ---------------------------------
                MAX_TEXT_LENGTH = 15000
                tender_text_for_ai = tender_text[:MAX_TEXT_LENGTH]

                if len(tender_text) > MAX_TEXT_LENGTH:
                    st.caption(
                        "For this prototype, only the first 15,000 characters "
                        "are sent to the AI analyzer."
                    )

                # ---------------------------------
                # AI Analysis
                # ---------------------------------
                with st.spinner(
                    "AI is analyzing the tender requirements..."
                ):
                    analysis = analyze_requirement(tender_text_for_ai, language)

                st.markdown("## AI Tender Understanding")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        f"""
                        <div class="section-card">
                            <b>Product</b><br>
                            {analysis.get("product", "Not identified")}
                            <br><br>

                            <b>Category</b><br>
                            {analysis.get("category", "Not identified")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    specifications = analysis.get("specifications", [])
                    if not isinstance(specifications, list):
                        specifications = [str(specifications)]

                    specs = ", ".join(specifications) if specifications else "Not identified"

                    st.markdown(
                        f"""
                        <div class="section-card">
                            <b>Application</b><br>
                            {analysis.get("application", "Not identified")}
                            <br><br>

                            <b>Specifications</b><br>
                            {specs}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                compliance_needs = analysis.get("compliance_needs", [])
                if not isinstance(compliance_needs, list):
                    compliance_needs = [str(compliance_needs)]

                compliance = ", ".join(compliance_needs)

                st.markdown(
                    f"""
                    <div class="section-card">
                        <b>Compliance Needs</b><br>
                        {compliance if compliance else "Not explicitly identified"}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---------------------------------
                # Semantic search
                # ---------------------------------
                search_query = analysis.get("search_query", "")

                if not search_query:
                    search_query = tender_text_for_ai

                results = search_standards(search_query)
                related_results = get_related_standards(results)

                st.session_state["latest_analysis"] = {
                    "source": "Tender PDF",
                    "query": tender_text_for_ai,
                    "analysis": analysis,
                    "results": results,
                    "related_results": related_results
                }

                st.markdown("## Recommended Standards")
                st.markdown(
                    '<span class="badge badge-blue">AI Retrieved</span>'
                    '<span class="badge badge-amber">Tender Context</span>',
                    unsafe_allow_html=True
                )

                if not results:

                    st.warning(
                        "No strong matching standards were found in the prototype dataset."
                    )

                else:

                    for i, result in enumerate(results, start=1):

                        st.markdown(
                            f"### {i}. {result['standard_id']} — {result['title']}"
                        )

                        st.markdown(
                            f"""
**Product:** {result['product']}

**Category:** {result['category']}

**Semantic Match:** {result['score']}%

**Certification:** {result['certification'] or 'Not available'}
"""
                        )

                        with st.expander(
                            f"Details for {result['standard_id']}"
                        ):

                            st.write(
                                "**Scope:**",
                                result["scope"]
                            )

                            st.write(
                                "**Related Standards:**",
                                result["related_standards"] or "Not available"
                            )

                    st.markdown("## Allied Standards")

                    if not related_results:
                        st.info(
                            "No additional allied standards were found "
                            "in the prototype knowledge base."
                        )
                    else:
                        for i, standard in enumerate(related_results, start=1):
                            st.markdown(
                                f"### {i}. {standard['standard_id']} — {standard['title']}"
                            )

                            st.markdown(
                                f"""
**Product:** {standard['product']}

**Category:** {standard['category']}

**Certification:** {standard['certification'] or 'Not available'}
"""
                            )

                            with st.expander(
                                f"View scope for {standard['standard_id']}"
                            ):
                                st.write(standard["scope"])

                    # ---------------------------------
                    # AI explanation
                    # ---------------------------------
                    with st.spinner(
                        "Generating AI recommendation explanation..."
                    ):
                        explanation = explain_recommendations(
                            tender_text_for_ai,
                            results,
                            language
                        )

                    explanation = clean_ai_output(explanation)

                    st.session_state["latest_analysis"]["explanation"] = explanation

                    st.markdown("## AI Recommendation Explanation")

                    st.markdown(
                        """
                        <div class="section-card">
                            <b>AI Explanation</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Render AI content as Markdown, not raw HTML.
                    st.markdown(explanation)

                    st.info(
                        "Prototype results are based on a demo standards knowledge base. "
                        "Final recommendations should be verified using official BIS sources."
                    )


# -------------------------------------------------
# STANDARDS EXPLORER
# -------------------------------------------------

elif page == "Standards Explorer":

    st.markdown(
        '<div class="main-title">Standards Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Browse the prototype standards knowledge base</div>',
        unsafe_allow_html=True
    )

    search_term = st.text_input(
        "Search by product, category, title, or keyword"
    )

    categories = ["All"] + sorted(
        df["category"].dropna().unique().tolist()
    )

    selected_category = st.selectbox(
        "Filter by Category",
        categories
    )

    filtered_df = df.copy()

    if selected_category != "All":

        filtered_df = filtered_df[
            filtered_df["category"] == selected_category
        ]

    if search_term:

        mask = (
            filtered_df["title"].str.contains(
                search_term,
                case=False,
                na=False
            )
            |
            filtered_df["product"].str.contains(
                search_term,
                case=False,
                na=False
            )
            |
            filtered_df["keywords"].str.contains(
                search_term,
                case=False,
                na=False
            )
        )

        filtered_df = filtered_df[mask]

    st.write(
        f"Showing **{len(filtered_df)}** standards"
    )

    display_columns = [
        "standard_id",
        "title",
        "product",
        "category",
        "certification"
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------
# COMPLIANCE SUMMARY
# -------------------------------------------------

elif page == "Compliance Summary":

    st.markdown(
        '<div class="main-title">Compliance Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-assisted procurement compliance overview</div>',
        unsafe_allow_html=True
    )

    if "latest_analysis" not in st.session_state:

        st.info(
            "Analyze a procurement requirement or tender first "
            "to generate a compliance summary."
        )

    else:

        data = st.session_state["latest_analysis"]
        analysis = data["analysis"]
        results = data["results"]
        related_results = data["related_results"]

        st.markdown("## Procurement Overview")
        st.markdown(
            '<span class="badge badge-teal">Latest Analysis</span>'
            '<span class="badge badge-blue">Explainable AI</span>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Primary Recommendations", len(results))
        col2.metric("Allied Standards", len(related_results))

        certification_count = sum(
            1
            for item in results + related_results
            if item.get("certification")
        )

        col3.metric("Certification Records", certification_count)

        st.markdown("## Product Information")

        st.write("**Source:**", data.get("source", "Not available"))
        st.write("**Product:**", analysis.get("product", "Not identified"))
        st.write("**Category:**", analysis.get("category", "Not identified"))
        st.write("**Application:**", analysis.get("application", "Not identified"))

        specifications = analysis.get("specifications", [])
        if isinstance(specifications, list):
            specifications = ", ".join(specifications)

        st.write(
            "**Specifications:**",
            specifications or "Not identified"
        )

        st.markdown("## Recommended Standards")

        if results:
            for result in results:
                st.markdown(
                    f"**{result['standard_id']} — {result['title']}**"
                )
                st.caption(
                    f"Semantic Match: {result['score']}%"
                )
        else:
            st.write("No primary recommendations available.")

        st.markdown("## Allied Standards")

        if related_results:
            for standard in related_results:
                st.markdown(
                    f"**{standard['standard_id']} — {standard['title']}**"
                )
                st.caption(
                    f"Category: {standard['category']}"
                )
        else:
            st.write("No additional allied standards identified.")

        st.markdown("## Compliance Indicators")

        if results:
            st.success("Relevant standards identified")

        if related_results:
            st.success("Related / allied standards identified")

        if certification_count > 0:
            st.success("Certification information available")


        st.markdown("## Download Report")

        explanation = data.get(
            "explanation",
            ""
        )

        pdf_report = generate_pdf_report(
            analysis=analysis,
            results=results,
            related_results=related_results,
            source=data.get(
                "source",
                "Requirement Analysis"
            ),
            explanation=explanation
        )

        st.download_button(
            label="Download PDF Compliance Report",
            data=pdf_report,
            file_name="BISense_AI_Compliance_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.warning(
            "Prototype recommendations must be verified "
            "against authoritative BIS sources before real procurement use."
        )


# -------------------------------------------------
# ABOUT
# -------------------------------------------------

elif page == "About":

    st.markdown(
        '<div class="main-title">About BISense AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        BISense AI is a prototype AI-powered recommendation system
        designed to assist procurement officials in identifying
        relevant Indian Standards from natural-language requirements.

        ### Core Capabilities

        - AI requirement understanding
        - Semantic standards search
        - Related standards discovery
        - Certification awareness
        - Explainable recommendations

        ### Current Prototype Stack

        **Frontend:** Streamlit  
        **AI:** Gemini  
        **Embeddings:** Sentence Transformers  
        **Semantic Search:** FAISS  
        **Data:** CSV-based standards knowledge base
        """
    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown(
    """
    <div class="footer-text">
        BISense AI • Internal Hackathon Prototype
    </div>
    """,
    unsafe_allow_html=True
)