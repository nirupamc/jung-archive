import traceback
from jung_archive.api.app import get_services, run_search
from jung_archive.api.schemas import SearchRequest

vi, bm25, reranker = get_services()

for mode in ['dense', 'bm25', 'hybrid', 'hybrid_rerank']:
    try:
        req = SearchRequest(query='How does Jung describe the Self?', mode=mode, top_k=5)
        resp = run_search(req)
        print(f'{mode.upper():20} results={len(resp.results)} warnings={resp.warnings}')
        for r in resp.results[:2]:
            title = str(r.title or '')[:40]
            print(f'    chunk={str(r.chunk_id)[:12]} doc={str(r.document_id)[:12]} title={title}')
    except Exception as e:
        print(f'{mode.upper():20} ERROR: {e}')
        traceback.print_exc()
    print()
