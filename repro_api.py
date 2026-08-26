from jung_archive.api.app import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)

# Test health
r = client.get('/api/health')
print('Health:', r.status_code, r.json())

# Test DENSE
r = client.post('/api/retrieval/search', json={'query': 'How does Jung describe the Self?', 'mode': 'dense', 'top_k': 5})
print('DENSE:', r.status_code)
if r.status_code != 200:
    print('  Body:', r.text[:500])
else:
    data = r.json()
    print('  results:', len(data.get('results', [])), 'warnings:', data.get('warnings', []))
    for res in data.get('results', [])[:3]:
        cid = str(res["chunk_id"])[:12]
        did = str(res["document_id"])[:12]
        title = str(res.get("title", ""))[:40]
        print(f'    chunk={cid} doc={did} title={title}')

# Test BM25
r = client.post('/api/retrieval/search', json={'query': 'How does Jung describe the Self?', 'mode': 'bm25', 'top_k': 5})
print('BM25:', r.status_code)
if r.status_code != 200:
    print('  Body:', r.text[:500])
else:
    data = r.json()
    print('  results:', len(data.get('results', [])), 'warnings:', data.get('warnings', []))

# Test HYBRID
r = client.post('/api/retrieval/search', json={'query': 'How does Jung describe the Self?', 'mode': 'hybrid', 'top_k': 5})
print('HYBRID:', r.status_code)
if r.status_code != 200:
    print('  Body:', r.text[:500])
else:
    data = r.json()
    print('  results:', len(data.get('results', [])), 'warnings:', data.get('warnings', []))

# Test HYBRID + RERANK
r = client.post('/api/retrieval/search', json={'query': 'How does Jung describe the Self?', 'mode': 'hybrid_rerank', 'top_k': 5})
print('HYBRID_RERANK:', r.status_code)
if r.status_code != 200:
    print('  Body:', r.text[:500])
else:
    data = r.json()
    print('  results:', len(data.get('results', [])), 'warnings:', data.get('warnings', []))
