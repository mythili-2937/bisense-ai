import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from ai_analyzer import analyze_requirement, explain_recommendations
from pathlib import Path


# -----------------------------
# Load dataset
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "standards.csv"

print("Dataset path:", CSV_PATH)
print("Dataset exists:", CSV_PATH.exists())

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} standards successfully!")


# -----------------------------
# Prepare text for embeddings
# -----------------------------

df["search_text"] = (
    df["title"].fillna("") + ". " +
    df["product"].fillna("") + ". " +
    df["category"].fillna("") + ". " +
    df["scope"].fillna("") + ". " +
    df["keywords"].fillna("")
)


# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["search_text"].tolist(),
    normalize_embeddings=True
)


# -----------------------------
# Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


# -----------------------------
# Semantic search function
# -----------------------------

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

        # Ignore weak matches
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
            "related_standards": row["related_standards"],
            "certification": row["certification"]
        })

    return results


# -----------------------------
# Main program
# -----------------------------

while True:

    query = input(
        "\nEnter procurement requirement (or type exit): "
    )

    if query.lower() == "exit":
        break

    print("\nAnalyzing requirement with AI...")

    # Step 1: Gemini understands the requirement
    analysis = analyze_requirement(query)

    print("\nAI Requirement Analysis")
    print("------------------------")

    print("Product:", analysis["product"])
    print("Category:", analysis["category"])

    print(
        "Specifications:",
        ", ".join(analysis["specifications"])
    )

    print(
        "Application:",
        analysis["application"]
    )

    print(
        "Compliance Needs:",
        ", ".join(analysis["compliance_needs"])
    )

    # Step 2: Use AI-generated search query
    search_query = analysis["search_query"]

    print("\nAI Search Query:")
    print(search_query)

    # Step 3: Semantic search
    results = search_standards(search_query)

    print("\nTop Recommended Standards")
    print("-------------------------")

    if not results:
        print("No strong matching standards found.")
        continue

    for i, result in enumerate(results, start=1):

        print(
            f"\n{i}. {result['standard_id']} - "
            f"{result['title']}"
        )

        print(
            f"   Product: {result['product']}"
        )

        print(
            f"   Category: {result['category']}"
        )

        print(
            f"   Semantic Match: "
            f"{result['score']}%"
        )

        print(
            f"   Related Standards: "
            f"{result['related_standards']}"
        )

        print(
            f"   Certification: "
            f"{result['certification']}"
        )

    # Step 4: Gemini explains retrieved results
    print("\nGenerating AI explanation...")

    explanation = explain_recommendations(
        query,
        results
    )

    print("\nAI Recommendation Explanation")
    print("-----------------------------")

    print(explanation)