import traceback
import sys

print("=== Step 1: Load embedding provider and embed query ===", flush=True)
try:
    from jung_archive.embedding.provider import LocalSentenceTransformerProvider
    provider = LocalSentenceTransformerProvider()
    vecs = provider.embed(["How does Jung describe the Self?"])
    print(f"Query embedded: shape={vecs.shape} dim={provider.dimension}", flush=True)
except Exception as e:
    print(f"Embed ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("=== Step 2: Chroma query ===", flush=True)
try:
    import chromadb
    client = chromadb.PersistentClient(path="data/chroma")
    col = client.get_collection("jung_archive")
    print(f"Collection count: {col.count()}", flush=True)
    print(f"Collection metadata: {col.metadata}", flush=True)
    
    # Query without filter
    res = col.query(
        query_embeddings=vecs.tolist(),
        n_results=5,
    )
    print(f"Query returned: {len(res['ids'][0])} results", flush=True)
    for i, cid in enumerate(res['ids'][0]):
        meta = res['metadatas'][0][i]
        dist = res['distances'][0][i]
        print(f"  [{i}] chunk={cid[:20]} dist={dist:.4f} doc={meta.get('document_id','')[:12]}", flush=True)
except Exception as e:
    print(f"Chroma ERROR: {e}", flush=True)
    traceback.print_exc()

print("=== Step 3: Chroma query with where filter ===", flush=True)
try:
    res2 = col.query(
        query_embeddings=vecs.tolist(),
        n_results=5,
        where={"document_id": {"$in": ["381d2da4b68e", "158b59f7a945"]}},
    )
    print(f"Query with filter returned: {len(res2['ids'][0])} results", flush=True)
    for i, cid in enumerate(res2['ids'][0]):
        meta = res2['metadatas'][0][i]
        dist = res2['distances'][0][i]
        print(f"  [{i}] chunk={cid[:20]} dist={dist:.4f} doc={meta.get('document_id','')[:12]}", flush=True)
except Exception as e:
    print(f"Chroma filter ERROR: {e}", flush=True)
    traceback.print_exc()
