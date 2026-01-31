"""
arXiv Paper Search

Auto-generated Streamlit app by Jotty V2.
Run with: streamlit run app.py
"""
import streamlit as st
import requests
import arxiv
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="arXiv Paper Search",
    page_icon="📚",
    layout="wide"
)

st.title("arXiv Paper Search")
st.markdown("Search and explore academic papers from arXiv.")


# Sidebar filters
with st.sidebar:
    st.header("Search Filters")
    search_query = st.text_input("Search Query", value="machine learning")
    max_results = st.slider("Max Results", 5, 50, 20)
    sort_by = st.selectbox("Sort By", ["relevance", "lastUpdatedDate", "submittedDate"])

    categories = st.multiselect(
        "Categories",
        ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.NE", "cs.IR"],
        default=["cs.AI", "cs.LG"]
    )

# Search button
if st.button("🔍 Search arXiv", type="primary") or search_query:
    with st.spinner("Searching arXiv..."):
        try:
            # Build query with categories
            if categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                full_query = f"({search_query}) AND ({cat_query})"
            else:
                full_query = search_query

            # Search arXiv
            sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }[sort_by]

            client = arxiv.Client()
            search = arxiv.Search(
                query=full_query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results = list(client.results(search))

            if results:
                st.success(f"Found {len(results)} papers")

                # Display results
                for i, paper in enumerate(results):
                    with st.expander(f"📄 {paper.title}", expanded=(i < 3)):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Authors:** {', '.join([a.name for a in paper.authors[:5]])}")
                            if len(paper.authors) > 5:
                                st.markdown(f"*...and {len(paper.authors) - 5} more*")

                            st.markdown(f"**Published:** {paper.published.strftime('%Y-%m-%d')}")
                            st.markdown(f"**Categories:** {', '.join(paper.categories)}")

                        with col2:
                            st.markdown(f"[PDF]({paper.pdf_url})")
                            st.markdown(f"[arXiv]({paper.entry_id})")

                            # Download PDF button
                            arxiv_id = paper.entry_id.split('/')[-1]
                            if st.button(f"⬇️ Download PDF", key=f"dl_{i}"):
                                with st.spinner("Downloading..."):
                                    try:
                                        response = requests.get(paper.pdf_url)
                                        if response.status_code == 200:
                                            st.download_button(
                                                label="💾 Save PDF",
                                                data=response.content,
                                                file_name=f"{arxiv_id}.pdf",
                                                mime="application/pdf",
                                                key=f"save_{i}"
                                            )
                                        else:
                                            st.error("Failed to fetch PDF")
                                    except Exception as e:
                                        st.error(f"Download error: {e}")

                        st.markdown("---")
                        st.markdown("**Abstract:**")
                        st.markdown(paper.summary)

                        # Citation
                        if st.button(f"📋 Copy BibTeX", key=f"cite_{i}"):
                            bibtex = f"""@article{{{paper.entry_id.split('/')[-1]},
    title={{{paper.title}}},
    author={{{' and '.join([a.name for a in paper.authors])}}},
    year={{{paper.published.year}}},
    journal={{arXiv preprint arXiv:{paper.entry_id.split('/')[-1]}}}
}}"""
                            st.code(bibtex, language="bibtex")

                # Summary table
                st.subheader("📊 Results Summary")
                df = pd.DataFrame([{
                    "Title": p.title[:60] + "..." if len(p.title) > 60 else p.title,
                    "Authors": ", ".join([a.name for a in p.authors[:2]]),
                    "Year": p.published.year,
                    "Categories": ", ".join(p.categories[:2]),
                } for p in results])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No papers found. Try different search terms.")

        except Exception as e:
            st.error(f"Search error: {e}")
            st.info("Make sure to install: pip install arxiv")

