# Fusion Retrieval and GraphRAG in ArcadeDB
## Algorithms, implementation patterns, ranking strategies, and an experimental design for an applied machine-learning thesis

**Implementation baseline:** ArcadeDB 26.11.1  
**Baseline date:** 14 November 2026  
**Scope:** Dense vector retrieval, learned-sparse retrieval, BM25 full-text search, graph retrieval, result fusion, reranking, and reproducible evaluation.

---

## Executive summary

Fusion retrieval combines ranked evidence from multiple retrieval systems to improve the relevance and coverage of a search result set. For a document-centric GraphRAG system, the most useful sources are:

1. **Dense vector retrieval** (`LSM_VECTOR`, `vector.neighbors`) for semantic similarity.
2. **Learned-sparse retrieval** (`LSM_SPARSE_VECTOR`, `vector.sparseNeighbors`) for token-level lexical/semantic matching, including SPLADE- or BGE-M3-style sparse representations.
3. **Full-text retrieval** (`FULL_TEXT`, `SEARCH_INDEX`, `SEARCH_FIELDS`) for BM25-ranked lexical matches.
4. **Graph retrieval** for explicit entity, concept, provenance, and relationship evidence.
5. **Reranking** for fine-grained relevance assessment of a bounded candidate set.

ArcadeDB 26.11.1 has native server-side fusion for dense, sparse, and full-text result sources through `vector.fuse()`. It supports Reciprocal Rank Fusion (RRF), Distribution-Based Score Fusion (DBSF), and LINEAR fusion, plus per-source weights, candidate limits, and grouping/diversification. The latest release also completes important Cypher 25 vector `SEARCH` behaviour and adds vector/full-text index DDL in openCypher.

**Key architectural recommendation:** use ArcadeDB's native `vector.fuse()` for the lexical/vector stage. Treat graph expansion as a separately measured retrieval leg, then either (a) use ArcadeDB's documented hybrid-search graph expansion where its constraints fit, or (b) export a bounded set of graph candidates with explicit provenance and fuse those candidates in a controlled application layer. Do not silently mix graph reachability with relevance: a short graph path is not necessarily evidence that a document answers a question.

The thesis should compare retrieval systems under the same corpus, embedding model, chunking, candidate budget, and evaluation queries. The central hypothesis is that graph-aware retrieval improves answer-supporting evidence for relationship- and multi-hop questions, while dense + BM25/sparse fusion improves lexical/semantic recall. These effects must be measured separately.

---

## 1. Terminology and scope

### 1.1 Query fusion, result fusion, and hybrid search

These terms are related but should not be treated as synonyms.

- **Query expansion/decomposition:** transform one user question into multiple retrieval queries or subquestions.
- **Multi-retriever / hybrid retrieval:** execute multiple retrieval strategies, such as dense vector and BM25 search.
- **Result fusion:** merge independently ranked result lists into one ranked list.
- **Reranking:** apply a more expensive relevance model to a bounded candidate set.
- **Graph expansion:** use retrieved entities or vertices as seeds to discover connected evidence through typed relationships.
- **Graph-aware fusion:** combine graph-derived evidence with document retrieval results, using explicit scores and provenance.

A system may use hybrid retrieval without query expansion. It may also use graph expansion after fusion, or treat graph retrieval as another candidate source. Those are different architectures and should be separate experimental conditions.

### 1.2 Why one retrieval method is insufficient

| Method | Strong at | Typical failure mode |
|---|---|---|
| Dense vectors | Paraphrases, semantic similarity, conceptual matching | Exact identifiers, rare terms, negation, and close lexical distinctions can be missed |
| BM25/full text | Names, identifiers, phrases, terminology, exact technical vocabulary | Vocabulary mismatch and paraphrases |
| Learned sparse vectors | Token-level term importance with expansion beyond literal BM25 | Requires a sparse encoder and consistent vocabulary/dimension mapping |
| Graph traversal | Explicit relationships, multi-hop context, entity-centric exploration | Graph incompleteness, noisy edges, hub expansion, and relation ambiguity |
| Cross-encoder reranker | Fine-grained query-passage relevance | More expensive; only sees candidates already retrieved |

The purpose of fusion is not to assume that every retriever is equally good. It is to combine complementary evidence and test whether the combination improves ranking.

---

## 2. ArcadeDB 26.11.1 retrieval capabilities

### 2.1 Relevant native indexes and functions

| Capability | ArcadeDB mechanism | Role |
|---|---|---|
| Dense vector | `LSM_VECTOR` | Approximate nearest-neighbour search |
| Dense retrieval | `vector.neighbors()` | Retrieve nearest records |
| Sparse vector | `LSM_SPARSE_VECTOR` | Learned sparse retrieval |
| Sparse retrieval | `vector.sparseNeighbors()` | Retrieve by sparse dot product |
| Full text | `FULL_TEXT` | Lucene-backed lexical index |
| Full-text retrieval | `SEARCH_INDEX()`, `SEARCH_FIELDS()` | BM25-ranked results |
| Native fusion | `vector.fuse()` | Combine two or more ranked result pipelines |
| Vector reranking | `vector.rerank()` | Rerank a fused candidate set using a dense vector |
| Graph traversal | SQL `TRAVERSE`, SQL graph functions, Cypher `MATCH` | Discover connected records |
| Cypher vector retrieval | Cypher 25 `SEARCH` | Bind vector-index candidates into graph patterns |
| Graph + hybrid API | Hybrid search endpoint / Studio hybrid search | Vector + full text + optional graph expansion |

ArcadeDB's dense vector implementation uses an LSM-backed index and JVector graph search. The current documentation describes Vamana as the flat/default structure and hierarchical HNSW as an available alternative. Dense similarity metrics include COSINE, DOT_PRODUCT, and EUCLIDEAN. For unit-normalized text embeddings, DOT_PRODUCT preserves cosine ranking and can avoid repeated normalization.

Sparse vectors are stored as parallel arrays of non-zero dimension IDs and weights. The arrays must have equal length; weights must be non-negative. Retrieval uses a Block-Max MaxScore strategy. This is not the same as BM25 full-text search: BM25 is supplied by the `FULL_TEXT` index, while `LSM_SPARSE_VECTOR` stores externally generated sparse embeddings.

### 2.2 Version-specific changes that matter

ArcadeDB 26.11.1 is the baseline for this report. Relevant changes include:

- Cypher 25 `SEARCH` now executes vector search and binds the nearest records; filtered search widens candidate exploration until it has the requested number of matching rows.
- Cypher index DDL supports `CREATE VECTOR INDEX` and `CREATE FULLTEXT INDEX`.
- `SEARCH_INDEX()` can be passed directly as a `vector.fuse()` source, including for score-based fusion; it cannot be bounded with a source-local `LIMIT`, so a subquery is preferable when the lexical result set could be large.
- `vector.rerank()` avoids reloading records already carried by upstream vector/fusion results and keeps only the best requested results while scanning.
- The default dense-search beam for graphs with at least 10,000 nodes changed to `max(2 × k, 100)`. Explicit `efSearch` remains important for controlled recall/latency experiments.
- Sparse postings are transactionally logged for crash recovery from this release.
- `PRODUCT` and `BINARY` vector index modes have candidate oversampling and reranking controls; they should be evaluated separately from unquantized or INT8 baselines.

Do not assume that examples written for older ArcadeDB releases have identical semantics. Pin the server and driver versions in the experiment.

---

## 3. Data model for a document corpus and knowledge graph

A useful thesis schema separates source documents, chunks, entities, concepts, and typed relationships. This avoids treating every extracted entity as if it were itself a retrievable passage.

### 3.1 Suggested vertex types

| Type | Important properties | Purpose |
|---|---|---|
| `SourceDocument` | `document_id`, `title`, `uri`, `source_type`, `published_at`, `checksum` | Original source and provenance |
| `Chunk` | `chunk_id`, `text`, `ordinal`, `token_count`, `embedding`, `tokens`, `weights`, `language` | Primary retrieval unit |
| `Entity` | `entity_id`, `canonical_name`, `entity_type`, `description` | Named entity |
| `Concept` | `concept_id`, `label`, `definition`, `domain` | Abstract concept |
| `Claim` | `claim_id`, `statement`, `confidence`, `extraction_method` | Atomic assertion extracted from corpus |

### 3.2 Suggested edge types

| Edge type | Direction / semantics | Suggested properties |
|---|---|---|
| `PART_OF` | Chunk → SourceDocument | `ordinal` |
| `MENTIONS` | Chunk → Entity | `mention_text`, `confidence`, `start_offset`, `end_offset` |
| `ABOUT` | Chunk → Concept | `confidence` |
| `RELATES_TO` | Entity/Concept → Entity/Concept | `relation_type`, `confidence`, `evidence_chunk_id` |
| `SUPPORTS` | Chunk → Claim | `confidence`, `evidence_span` |
| `CONTRADICTS` | Chunk → Claim | `confidence`, `evidence_span` |
| `SAME_AS` | Entity → Entity | `confidence`, `resolution_method` |
| `DEPENDS_ON` | Concept/Entity → Concept/Entity | `confidence`, `evidence_chunk_id` |

Store the evidence chunk or source reference on extracted relationships. A graph edge without provenance is difficult to audit and should not be treated as equivalent to a source passage.

### 3.3 Chunk-first retrieval

For RAG, retrieve `Chunk` records, not only `Entity` vertices. Entities and concepts are useful as graph navigation points, but the LLM ultimately needs passages that contain evidence.

A recommended path is:

`Question → retrieved chunks → entity/concept anchors → bounded graph expansion → supporting chunks → fusion/reranking → context`

This keeps graph context connected to textual evidence.

### 3.4 Embedding storage and external properties

ArcadeDB documentation recommends considering `EXTERNAL true` for large embedding properties when most graph queries do not read the embedding. This can prevent large vectors from displacing graph topology in the page cache. It is a workload-dependent choice: if nearly every operation reads the embedding, the extra indirection may not help.

For example, for a 1,024-dimensional float embedding:

```sql
CREATE PROPERTY Chunk.embedding ARRAY_OF_FLOATS (EXTERNAL true);

CREATE INDEX ON Chunk (embedding) LSM_VECTOR METADATA {
  dimensions: 1024,
  similarity: 'COSINE'
};
```

Use the actual dimension emitted by the selected embedding model. Do not infer it from a model name or silently truncate/pad vectors.

---

## 4. Index creation and retrieval

The following examples use SQL syntax documented for ArcadeDB. Treat names and dimensions as a template to adapt to the actual schema. Run DDL and queries against the pinned 26.11.1 server and validate with Studio/EXPLAIN before using them in the benchmark.

### 4.1 Dense vector index

```sql
CREATE INDEX ON Chunk (embedding) LSM_VECTOR METADATA {
  dimensions: 1024,
  similarity: 'COSINE'
};
```

Retrieve the nearest 50 chunks:

```sql
SELECT expand(
  vector.neighbors('Chunk[embedding]', :queryVector, 50)
);
```

For a recall-sensitive query, set the search beam explicitly:

```sql
SELECT expand(
  vector.neighbors(
    'Chunk[embedding]',
    :queryVector,
    50,
    { efSearch: 400 }
  )
);
```

The `efSearch` parameter controls the graph-search beam. Higher values generally improve recall at the cost of latency. For experiments, record the value per query and do not compare configurations that silently use different defaults.

### 4.2 Filtered dense retrieval

A tenant, document class, language, or access-control filter must be applied consistently to every retrieval leg. One documented pattern is an allowed-RID filter:

```sql
SELECT expand(
  vector.neighbors(
    'Chunk[embedding]',
    :queryVector,
    50,
    {
      filter: (
        SELECT @rid
        FROM Chunk
        WHERE language = 'en'
          AND corpus_id = :corpusId
      )
    }
  )
);
```

Do not use a post-retrieval application filter as a substitute for access control. If the user is not authorised to read a chunk, it must not enter the candidate set or be exposed through graph expansion.

ArcadeDB also documents partition-aware vector search. Where the data model uses a partitioned bucket strategy and the query constrains the partition key, the planner can target the relevant per-bucket vector graph. This is a schema-level optimisation, not a reason to omit logical authorisation checks.

### 4.3 Full-text index with BM25

```sql
CREATE INDEX ON Chunk (text) FULL_TEXT;
```

Search by Lucene syntax:

```sql
SELECT *
FROM Chunk
WHERE SEARCH_INDEX('Chunk[text]', :luceneQuery);
```

`FULL_TEXT` uses BM25 by default for newly created indexes. Results can be ordered by `$score DESC`. A multi-property index can cover title and body-like fields:

```sql
CREATE INDEX ON Chunk (title, text) FULL_TEXT;
```

Then use `SEARCH_FIELDS()` to restrict the search to the intended fields:

```sql
SELECT *
FROM Chunk
WHERE SEARCH_FIELDS(['title', 'text'], :luceneQuery);
```

Use Lucene query syntax deliberately. Escape or construct user-provided query text safely in the application; do not concatenate untrusted text into SQL.

### 4.4 Learned-sparse index

If the ingestion pipeline produces sparse token IDs and corresponding weights (for example, using BGE-M3's learned sparse output), store them as parallel arrays:

```sql
CREATE PROPERTY Chunk.tokens ARRAY_OF_INTEGERS;
CREATE PROPERTY Chunk.weights ARRAY_OF_FLOATS;

CREATE INDEX ON Chunk (tokens, weights) LSM_SPARSE_VECTOR
  METADATA {
    "dimensions": 0,
    "modifier": "IDF"
  };
```

Here `dimensions: 0` means an open-ended vocabulary cap in the documented syntax. Confirm the exact supported metadata and model output contract for the selected model.

Retrieve sparse candidates:

```sql
SELECT expand(
  vector.sparseNeighbors(
    'Chunk[tokens,weights]',
    :queryIndices,
    :queryWeights,
    50
  )
);
```

The query indices and weights must align positionally. Validate equal lengths, integer dimension IDs, non-negative weights, and consistent model vocabulary/version during ingestion.

### 4.5 Candidate pool sizing

A useful starting point for a final top-10 result is to fetch 50–100 candidates per leg, fuse to 50–100 candidates, then rerank to 10. These are initial experimental values, not universal optima.

Candidate recall is a hard upper bound on final ranking quality: a reranker cannot recover a relevant passage that none of the retrieval legs retrieved.

---

## 5. Fusion algorithms

### 5.1 Reciprocal Rank Fusion (RRF)

RRF combines ranking positions rather than raw similarity scores:

\[
RRF(d) = \sum_{i=1}^{m} \frac{w_i}{k + r_i(d)}
\]

Where:

- \(d\) is a candidate document/chunk;
- \(m\) is the number of retrieval lists;
- \(r_i(d)\) is the one-based rank of \(d\) in list \(i\);
- \(w_i\) is the weight of source \(i\);
- \(k\) is a smoothing constant.

ArcadeDB's documented default is `k = 60`.

**Advantages**
- Does not require cosine distance, BM25, and sparse scores to share a numeric scale.
- Robust when retrievers produce very different score distributions.
- Simple to explain and reproduce.
- A good first baseline for dense + lexical fusion.

**Limitations**
- Uses rank, not the magnitude of the original score.
- A small rank difference can matter even when the source scores are nearly tied.
- It cannot distinguish a very strong match from a marginal match if their rank positions are similar.
- Weights and candidate depths still require tuning.

RRF should be the primary fusion baseline in the thesis because it avoids a common methodological error: directly adding cosine similarity to BM25 scores.

### 5.2 Distribution-Based Score Fusion (DBSF)

DBSF normalises scores within each result distribution, using a range around the mean and standard deviation, then combines the normalised scores.

**Advantages**
- Retains some information about score magnitude.
- Can work when the score distributions are reasonably well behaved.
- Useful as a comparison to rank-only fusion.

**Limitations**
- Sensitive to the shape and size of each candidate distribution.
- Distribution estimates can be unstable for small or highly skewed result sets.
- Scores from different retrievers may not represent equivalent notions of relevance.

Use DBSF as an experimental condition, not as an assumed improvement.

### 5.3 LINEAR fusion

LINEAR fusion min-max normalises each source's scores and combines them with weights:

\[
S_{\mathrm{linear}}(d) =
\sum_{i=1}^{m} w_i \hat{s}_i(d)
\]

where \(\hat{s}_i(d)\) is the normalised score for source \(i\).

**Advantages**
- Intuitive weights.
- Score magnitude influences the result.
- Useful when weights have been tuned on a validation set.

**Limitations**
- Min-max normalisation is sensitive to outliers and candidate-set composition.
- Adding or removing candidates can change the normalised score of an existing candidate.
- Requires numeric scores with consistent direction (`higher = better`).
- Dense distance values must be converted to similarity; ArcadeDB documents that `vector.fuse()` automatically flips the dense `distance` field when extracting `vector.neighbors()` results.

Do not tune weights on the test set. Use a held-out validation set or nested evaluation.

### 5.4 Graph-aware scoring

Graph evidence is not naturally equivalent to vector similarity or BM25. A graph candidate may be relevant because it is connected to a query entity, not because it has a high semantic score.

A thesis can test explicit graph features such as:

- shortest path length from a query entity;
- relationship-type relevance;
- edge confidence;
- number of independent supporting source chunks;
- provenance quality;
- whether the path is supported by an actual source passage;
- entity-resolution confidence;
- graph degree / hubness penalty.

An illustrative application-layer score is:

\[
S_g(c,q) =
\alpha \frac{1}{1 + \ell(c,q)}
+ \beta C_{\mathrm{edge}}(c,q)
+ \gamma C_{\mathrm{provenance}}(c)
- \delta H(c)
\]

where \(\ell\) is path length, \(C_{\mathrm{edge}}\) is path confidence, \(C_{\mathrm{provenance}}\) is evidence quality, and \(H\) is a hubness penalty.

This is a proposed research feature, not a built-in ArcadeDB ranking formula. It must be validated and calibrated.

A safer alternative for the first experiment is to use graph traversal only to generate a bounded candidate list, then apply RRF to graph-derived ranks and the vector/full-text ranks. That tests the value of graph expansion without claiming that path length is a calibrated relevance score.

---

## 6. Native ArcadeDB fusion with `vector.fuse()`

### 6.1 Dense + sparse

```sql
SELECT expand(
  vector.fuse(
    vector.neighbors('Chunk[embedding]', :denseVector, 100),
    vector.sparseNeighbors(
      'Chunk[tokens,weights]',
      :queryIndices,
      :queryWeights,
      100
    ),
    { fusion: 'RRF', k: 60 }
  )
)
LIMIT 10;
```

This is the recommended first native hybrid baseline when both dense and learned-sparse representations are available.

### 6.2 Dense + full text

```sql
SELECT expand(
  vector.fuse(
    vector.neighbors('Chunk[embedding]', :denseVector, 100),
    (
      SELECT @rid, $score
      FROM Chunk
      WHERE SEARCH_INDEX('Chunk[text]', :luceneQuery) = true
      LIMIT 100
    ),
    { fusion: 'RRF', k: 60 }
  )
)
LIMIT 10;
```

The full-text subquery returns the record identifier and BM25 score. ArcadeDB documents that a `SEARCH_INDEX` subquery without an `ORDER BY` returns matches in descending score order, so the `LIMIT` retains the top lexical matches.

### 6.3 Dense + sparse + full text

```sql
SELECT expand(
  vector.fuse(
    vector.neighbors('Chunk[embedding]', :denseVector, 100),
    vector.sparseNeighbors(
      'Chunk[tokens,weights]',
      :queryIndices,
      :queryWeights,
      100
    ),
    (
      SELECT @rid, $score
      FROM Chunk
      WHERE SEARCH_INDEX('Chunk[text]', :luceneQuery) = true
      LIMIT 100
    ),
    { fusion: 'RRF', k: 60 }
  )
)
LIMIT 10;
```

This is a strong three-leg baseline for a corpus where dense semantic matching, learned sparse matching, and BM25 are all available.

### 6.4 Weighted RRF

```sql
SELECT expand(
  vector.fuse(
    vector.neighbors('Chunk[embedding]', :denseVector, 100),
    vector.sparseNeighbors(
      'Chunk[tokens,weights]',
      :queryIndices,
      :queryWeights,
      100
    ),
    {
      fusion: 'RRF',
      k: 60,
      weights: [2.0, 1.0]
    }
  )
)
LIMIT 10;
```

Weights correspond to the source order. The example gives dense retrieval twice the influence of sparse retrieval. It is illustrative only; estimate weights from validation data.

### 6.5 DBSF and LINEAR

The same source pattern can be used with:

```sql
{ fusion: 'DBSF' }
```

or:

```sql
{ fusion: 'LINEAR', weights: [0.6, 0.4] }
```

For LINEAR and DBSF, custom sources must expose a numeric `score` with higher meaning better. RRF is rank-based and is more tolerant of heterogeneous score scales.

### 6.6 Candidate limits and reranking

Bound every leg. An unbounded full-text result list can dominate query time and memory even if the final output is only ten records.

A typical two-stage pipeline is:

```sql
SELECT expand(
  vector.rerank(
    vector.fuse(
      vector.neighbors('Chunk[embedding]', :denseVector, :fetch),
      (
        SELECT @rid, $score
        FROM Chunk
        WHERE SEARCH_INDEX('Chunk[text]', :luceneQuery) = true
        LIMIT :fetch
      ),
      { fusion: 'RRF', limit: :candidateLimit }
    ),
    :denseVector,
    'dense',
    :finalK
  )
);
```

This example illustrates dense-vector reranking after fusion. It does **not** replace a cross-encoder reranker. A cross-encoder evaluates query-passage pairs jointly and can be implemented in the application layer after retrieving the candidate records.

Suggested initial values:

| Parameter | Initial value | Purpose |
|---|---:|---|
| `fetch` | 100 | Candidates from each source |
| `candidateLimit` | 50 | Fused candidates passed to reranking |
| `finalK` | 10 | Final retrieved passages |

Tune these values against recall, latency, and memory.

### 6.7 Diversification by source document

Chunk-level retrieval can return many near-duplicate chunks from one long document. ArcadeDB's `groupBy` and `groupSize` options can diversify retrieval at the index stage:

```sql
SELECT expand(
  vector.neighbors(
    'Chunk[embedding]',
    :denseVector,
    10,
    { groupBy: 'source_document_id', groupSize: 1 }
  )
);
```

This requests up to ten distinct source-document groups, keeping one chunk per group. Test diversification independently because it can improve source coverage while lowering raw passage-level relevance.

---

## 7. Graph retrieval and graph expansion

### 7.1 Graph retrieval should be seeded

Do not traverse the whole graph for every question. First retrieve query-relevant chunks and/or resolve query entities. Use those as seeds for a bounded traversal.

For example, if a query resolves to an `Entity` vertex, a bounded traversal can discover related entities and concepts:

```sql
SELECT FROM (
  TRAVERSE out('RELATES_TO', 'DEPENDS_ON', 'MENTIONS')
  FROM (
    SELECT FROM Entity
    WHERE canonical_name = :entityName
  )
  MAXDEPTH 2
  STRATEGY BREADTH_FIRST
)
WHERE @type IN ['Entity', 'Concept', 'Chunk'];
```

This is a conceptual pattern. Adapt edge direction and types to the actual schema. If the relationship is stored in the opposite direction, traverse the appropriate incoming or bidirectional edges. Keep the traversal depth small and filter the returned records.

### 7.2 Why breadth-first traversal matters

ArcadeDB's SQL `TRAVERSE` supports `MAXDEPTH`, `WHILE`, `LIMIT`, and depth-first or breadth-first strategies. For a seed-based neighbourhood search where hop distance is a feature, breadth-first traversal is generally easier to interpret because the first discovery depth corresponds to the shortest hop distance.

Important details:

- Put `MAXDEPTH` inside the traversal rather than filtering depth only in an outer query.
- Use explicit edge types rather than `out()` over every relationship.
- Avoid traversing all properties (`*`) unless required.
- Bound the number of returned records.
- Track path and provenance where the query requires an explanation.
- Be alert to high-degree vertices that can cause candidate explosion.

### 7.3 Cypher 25 vector `SEARCH`

ArcadeDB 26.11.1 supports the Cypher 25 `SEARCH` clause for vector retrieval. An index can be declared as follows:

```cypher
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};
```

A vector search with a property filter:

```cypher
MATCH (c:Chunk)
SEARCH c IN (
  VECTOR INDEX chunk_embedding
  FOR $queryVector
  WHERE c.language = 'en'
  LIMIT 20
) SCORE AS score
RETURN c.chunk_id AS chunkId, c.text AS text, score
ORDER BY score DESC;
```

This is particularly useful when the retrieval operation should be expressed alongside graph pattern matching. The `SEARCH` clause runs vector retrieval first and binds the selected nodes. In 26.11.1 the inner filter is tested against candidates while search widens to obtain the requested number of matches.

### 7.4 Full-text retrieval in Cypher

ArcadeDB documents a Cypher full-text procedure:

```cypher
CALL db.index.fulltext.queryNodes(
  'Chunk[text]',
  $luceneQuery,
  { limit: 100 }
)
YIELD node, score
RETURN node, score
ORDER BY score DESC;
```

This makes it possible to combine full-text retrieval with graph pattern matching in Cypher. Use the actual index name shown by the database.

### 7.5 Native hybrid search with graph expansion

ArcadeDB's Studio hybrid-search form exposes a vector leg, a full-text leg, a fusion strategy, per-leg weights, and optional graph expansion with edge types, direction, and maximum depth. The gRPC HybridSearch API documents corresponding fields:

- vector index and query vector;
- optional full-text index and Lucene query;
- fusion strategy: RRF, DBSF, or LINEAR;
- weights for `vector`, `fulltext`, and `expand`;
- optional graph expansion with edge types, direction (`out`, `in`, `both`), and `max_depth` from 1 to 3;
- the graph expansion leg requires RRF.

This is the most direct native route to test vector + full-text + graph expansion. Validate the HTTP/gRPC request schema for the exact 26.11.1 server and client before writing production integration code. The Studio UI and API are useful for a reproducible native baseline, but the thesis should also preserve raw requests and parameters.

### 7.6 Graph candidates as a separate retrieval leg

For more control over the research design, use an explicit graph leg:

1. Retrieve top query entities/concepts using exact lookup, dense retrieval, or full-text.
2. Traverse a fixed set of edge types to a bounded depth.
3. Map reached entities/concepts back to chunks using `MENTIONS`, `ABOUT`, `SUPPORTS`, or other evidence edges.
4. Deduplicate chunk RIDs.
5. Assign a graph rank or an explicit graph score.
6. Fuse graph candidates with dense and lexical candidates.
7. Preserve the graph path and supporting chunk IDs as provenance.

This can be implemented as separate ArcadeDB queries followed by application-layer fusion, or as a database query pipeline where a graph-derived subquery returns `@rid` and a score. Confirm the exact result shape accepted by `vector.fuse()` before treating a custom graph subquery as a supported source.

**Research advantage:** the graph leg becomes independently switchable, measurable, and ablatable.

---

## 8. Query decomposition and expansion

Query expansion is distinct from result fusion. It generates multiple queries before retrieval.

For a question such as:

> How does an application establish agent identity and retain human accountability for agent actions?

Possible subqueries:

1. agent identity and registration;
2. authentication and authorisation of autonomous agents;
3. mapping agents to human owners;
4. audit logs and responsibility attribution;
5. policy enforcement and access delegation.

Run the original query and the decomposed queries through one or more retrievers. Merge the resulting lists using RRF or another chosen fusion strategy.

### 8.1 Avoid uncontrolled expansion

Expansion can reduce precision by introducing loosely related terms. Apply these controls:

- retain the original query as a retrieval leg;
- limit the number of generated subqueries;
- require each subquery to represent a distinct information need;
- log the generated queries;
- compare expansion enabled/disabled;
- use a validation set to tune query count and weights;
- avoid allowing the LLM to invent unsupported domain facts during query generation.

### 8.2 Query intent classification

A useful extension is to classify questions into retrieval intents:

| Intent | Retrieval emphasis |
|---|---|
| Definition / explanation | Dense + BM25 |
| Exact name, identifier, standard, API | BM25 + exact lookup |
| Entity relationship | Entity resolution + graph |
| Multi-hop question | Graph + dense/BM25 evidence |
| Comparison | Query decomposition + diversified retrieval |
| Provenance / source verification | Graph provenance + source metadata |

This is a routing policy, not a replacement for a common baseline. Evaluate it against a fixed hybrid pipeline to determine whether routing improves quality enough to justify complexity.

---

## 9. Recommended reference architecture

```text
                 User question
                       |
             Normalisation / intent
                       |
          Optional query decomposition
                       |
       +---------------+----------------+
       |               |                |
   Dense query      BM25 query      Sparse query
       |               |                |
 vector.neighbors  SEARCH_INDEX   sparseNeighbors
       |               |                |
       +---------------+----------------+
                       |
                 vector.fuse()
                       |
                Candidate set
                       |
          Entity / concept resolution
                       |
             Bounded graph expansion
                       |
             Graph evidence candidates
                       |
        Graph-aware merge / deduplication
                       |
            Cross-encoder reranking
                       |
        Source-aware context construction
                       |
                 LLM generation
                       |
        Answer + citations / provenance
```

Two valid execution orders should be compared:

**Architecture A — retrieve, fuse, then expand graph**

- Dense/BM25/sparse fusion finds relevant chunks.
- Entities are extracted/resolved from the retrieved chunks.
- Graph expansion enriches those chunks with connected evidence.
- Reranking and context assembly follow.

This is generally efficient and reduces graph traversal to a smaller seed set.

**Architecture B — parallel graph and text retrieval**

- Resolve query entities independently.
- Run graph retrieval in parallel with vector and lexical retrieval.
- Fuse all candidate chunks using a controlled ranking policy.
- Rerank and assemble context.

This can improve multi-hop recall when relevant graph anchors are not mentioned in the top initial chunks, but may increase latency and noise.

**Recommendation:** implement Architecture A first. Add Architecture B as an experimental condition, not as the initial production default.

---

## 10. Ranking and reranking strategy

### 10.1 Recommended ranking cascade

| Stage | Method | Output |
|---|---|---|
| Candidate retrieval | Dense + BM25 (and sparse where available) | Per-source ranked lists |
| First fusion | RRF | Unified candidate ranking |
| Graph enrichment | Bounded typed traversal | Related evidence and paths |
| Graph merge | RRF or calibrated graph features | Graph-aware candidate pool |
| Reranking | Cross-encoder | Fine-grained query-passage ranking |
| Diversification | Source-level cap / MMR-style policy | Less redundancy |
| Context selection | Token-budget and provenance constraints | Final evidence set |

Do not add every technique at once. The experimental design should isolate each contribution.

### 10.2 Cross-encoder reranking

A cross-encoder takes a query and candidate passage together and predicts relevance. It is more computationally expensive than bi-encoder retrieval but can assess token-level interactions.

A practical design:

- retrieve 50–100 candidates from each source;
- fuse to 50–100 unique chunks;
- apply a cross-encoder to the fused pool;
- select the final 5–15 chunks, subject to a context token budget.

Measure reranker latency separately from retrieval latency. Use the same reranker across the compared retrieval conditions.

### 10.3 Graph-aware reranking features

For each candidate chunk, derive features such as:

- best graph path length to a query entity;
- number of distinct query entities connected to the chunk;
- edge-type sequence;
- mean/minimum edge confidence;
- number of distinct supporting source documents;
- entity-resolution confidence;
- graph hubness / degree;
- whether the graph path is backed by a source chunk.

Start with a transparent linear model or feature-based reranker. Avoid a complex learned ranker until there are enough judged query-document pairs.

### 10.4 Evidence quality and contradiction

A graph edge may represent an extracted assertion rather than a verified fact. Preserve:

- source document and chunk;
- extraction model and prompt/version;
- extraction confidence;
- entity-resolution method and confidence;
- timestamp/version;
- whether the relation is asserted, inferred, or manually verified.

Contradictory claims should not be collapsed into a single edge without preserving their source and polarity. Retrieval should be able to surface conflicting evidence when the question asks for a comparison or when the corpus itself is inconsistent.

---

## 11. Experimental design for the master's thesis

### 11.1 Research objective

Determine whether combining vector, lexical, and graph retrieval improves retrieval effectiveness and downstream answer quality over vector-only and conventional hybrid baselines in a document corpus represented in ArcadeDB.

### 11.2 Research questions

**RQ1 — Hybrid retrieval:** Does dense + BM25 fusion outperform dense-only retrieval?

**RQ2 — Learned sparse retrieval:** Does adding a learned-sparse leg improve recall and ranking quality beyond dense + BM25?

**RQ3 — Graph contribution:** Does graph expansion improve retrieval for entity-centric and multi-hop questions?

**RQ4 — Fusion strategy:** How do RRF, DBSF, and LINEAR compare under controlled candidate budgets?

**RQ5 — Query expansion:** Does query decomposition improve retrieval, and does it interact positively or negatively with graph retrieval?

**RQ6 — Efficiency:** What are the latency, memory, storage, and ingestion costs of each retrieval architecture?

**RQ7 — Answer quality:** Do retrieval improvements lead to more grounded, complete, and faithful LLM answers?

### 11.3 Hypotheses

- **H1:** Dense + BM25 RRF improves Recall@K over dense-only retrieval.
- **H2:** Learned-sparse retrieval adds value for terminology-heavy and exact-concept queries.
- **H3:** Graph expansion improves recall and answer completeness on multi-hop/entity-relationship questions more than on simple definitional questions.
- **H4:** RRF is a strong robust baseline when score distributions differ; score-based fusion may outperform it after validation-set calibration.
- **H5:** Graph expansion increases candidate recall but can reduce precision unless traversal depth, edge types, and hubness are controlled.
- **H6:** A cross-encoder improves ranking quality on the fused candidate pool, provided candidate recall is sufficient.
- **H7:** Retrieval quality gains may come with latency and indexing costs; the best architecture depends on the quality/latency trade-off.

These are hypotheses to test, not expected outcomes to report as findings.

### 11.4 Dataset and corpus preparation

Use a corpus with a stable snapshot and record:

- source count and source types;
- total extracted text size;
- chunk count;
- chunking method and overlap;
- language distribution;
- document metadata coverage;
- entity/concept count;
- edge count by relationship type;
- graph density and degree distribution;
- duplicate and near-duplicate rates;
- embedding model and version;
- sparse model and vocabulary/version;
- extraction model, prompt, and version.

Docling should be part of the reproducible ingestion pipeline. Preserve the original file, Docling output, chunk boundaries, metadata, and provenance mapping. A graph edge must be traceable back to the source text from which it was inferred.

### 11.5 Query benchmark

Create a human-reviewed query set with distinct categories:

| Query category | Example form | Primary capability tested |
|---|---|---|
| Semantic paraphrase | “What controls whether an agent may act?” | Dense retrieval |
| Exact terminology | “What is the purpose of an Entra service principal?” | BM25 / exact vocabulary |
| Rare identifier | Specific API, standard, or version | Lexical retrieval |
| Entity relation | “Which component authenticates this agent?” | Graph relationship |
| Multi-hop | “Which human is accountable for actions by an agent deployed through this application?” | Graph traversal |
| Comparative | “How do the two identity models differ?” | Decomposition and diversified evidence |
| Provenance | “Which source supports this requirement?” | Evidence and provenance |
| Contradiction | “Do the sources agree on this requirement?” | Conflicting evidence retrieval |

A reasonable initial target is 150–300 queries, with at least 20–30 per major category if resources permit. The final number should be driven by corpus size and annotation capacity. Split by topic or source where possible to reduce leakage.

### 11.6 Relevance judgements

For each query, annotate a pool of candidate chunks with graded relevance:

| Grade | Meaning |
|---:|---|
| 0 | Irrelevant |
| 1 | Topically related but does not answer the question |
| 2 | Partially useful evidence |
| 3 | Directly relevant / answer-supporting evidence |

Use at least two annotators for a subset, measure agreement (for example, weighted Cohen's kappa for two annotators), adjudicate disagreements, and publish the annotation protocol. If only one annotator is available, disclose this limitation and perform a second-pass consistency review.

For multi-hop questions, annotate not only individual passages but also whether the retrieved evidence set collectively supports the required reasoning chain.

### 11.7 Retrieval conditions / ablation matrix

Keep the embedding model, chunking, corpus, and query set fixed.

| ID | Configuration | Purpose |
|---|---|---|
| A | Dense only | Semantic baseline |
| B | BM25 only | Lexical baseline |
| C | Dense + BM25, RRF | Conventional hybrid baseline |
| D | Dense + sparse, RRF | Learned-sparse contribution |
| E | Dense + BM25 + sparse, RRF | Three-way lexical/semantic fusion |
| F | Dense + BM25, DBSF | Score-distribution fusion |
| G | Dense + BM25, LINEAR | Calibrated score fusion |
| H | Dense + BM25, RRF + graph expansion | Graph contribution |
| I | Dense + BM25 + sparse, RRF + graph | Full hybrid GraphRAG |
| J | I + cross-encoder reranking | Reranking contribution |
| K | J + query decomposition | Query expansion contribution |
| L | Graph-first / parallel graph and text retrieval | Execution-order comparison |

Not every condition must be implemented in the first iteration. A staged experiment is more manageable:

1. A–C establish dense and lexical baselines.
2. D–G study sparse retrieval and fusion algorithms.
3. H–I isolate graph value.
4. J–K measure reranking and query expansion.
5. L studies architecture ordering if time allows.

### 11.8 Retrieval metrics

Use standard ranking metrics:

**Recall@K**

\[
Recall@K = \frac{|R_q \cap Retrieved_q^K|}{|R_q|}
\]

Measures the fraction of known relevant chunks retrieved in the top K.

**Precision@K**

\[
Precision@K = \frac{|R_q \cap Retrieved_q^K|}{K}
\]

Measures the proportion of top-K results judged relevant.

**Mean Reciprocal Rank (MRR)**

\[
MRR = \frac{1}{|Q|}\sum_{q \in Q}\frac{1}{rank_q}
\]

Uses the rank of the first relevant result.

**nDCG@K**

Uses graded relevance and rewards highly relevant results appearing near the top. This is particularly suitable for the proposed 0–3 relevance scale.

Also report:

- MAP@K where appropriate;
- candidate recall before reranking;
- unique source-document coverage;
- duplicate chunk rate;
- graph-path evidence coverage for multi-hop questions;
- proportion of retrieved graph facts with source provenance.

Do not report only a single aggregate score. Break results down by query category.

### 11.9 Answer-level metrics

Retrieval quality is not identical to answer quality. For a downstream RAG evaluation, measure:

- answer correctness against a reference answer;
- completeness of required answer elements;
- faithfulness to retrieved evidence;
- citation precision (whether cited passages support the associated claims);
- citation recall (whether material claims have supporting citations);
- unsupported-claim rate;
- abstention behaviour when evidence is insufficient.

Use human evaluation for a carefully sampled subset. Automated LLM-as-judge scoring can help scale evaluation, but it should not be the sole ground truth. Blind evaluators to the retrieval condition where feasible.

### 11.10 Performance and resource metrics

Measure:

| Metric | Measurement |
|---|---|
| Dense retrieval latency | Per-query p50, p95, p99 |
| Full-text latency | Per-query p50, p95, p99 |
| Sparse retrieval latency | Per-query p50, p95, p99 |
| Fusion latency | Incremental time for merge |
| Graph traversal latency | Per-query, with seed count/depth |
| Reranking latency | Total and per candidate |
| End-to-end retrieval latency | Before LLM generation |
| Peak memory / heap | Under fixed load |
| Index storage | Dense, sparse, full-text separately |
| Ingestion throughput | Chunks/s and edges/s |
| Index build time | Per index |
| Update cost | Inserts, updates, deletes |
| LLM generation latency | Report separately from retrieval |

Use warm-up runs and repeated trials. Report p50 and p95, not only the mean. Keep concurrency and cache conditions controlled. Distinguish cold-start index loading from steady-state search.

### 11.11 Statistical analysis

For each query, compare paired metric values across configurations. Use:

- paired bootstrap confidence intervals for Recall@K, nDCG@K, and answer-level metrics;
- paired permutation tests or Wilcoxon signed-rank tests where appropriate;
- effect sizes, not only p-values;
- multiple-comparison correction if many configurations are tested.

Stratify by query type. A system can improve multi-hop retrieval while having little effect on simple fact lookups; an aggregate average can hide this.

### 11.12 Reproducibility controls

Record and pin:

- ArcadeDB server image/version and configuration;
- database schema and index metadata;
- driver and client versions;
- Docling version and parsing configuration;
- embedding and sparse model names, revisions, dimensions, and preprocessing;
- chunking algorithm and parameters;
- graph extraction prompt/model/version;
- entity-resolution rules;
- fusion parameters and candidate limits;
- reranker version;
- query benchmark and relevance labels;
- hardware, operating system, JVM, heap, and storage;
- warm-up procedure, concurrency, and number of repetitions.

Persist each experiment's raw ranked lists, per-source scores, fused ranks, graph paths, reranker scores, and final context. Without raw retrieval traces, a result is difficult to reproduce or diagnose.

---

## 12. Implementation plan

### Phase 1 — Corpus and baseline

1. Freeze a representative corpus snapshot.
2. Validate Docling parsing and chunk provenance.
3. Create `Chunk`, `SourceDocument`, `Entity`, and `Concept` types.
4. Generate dense embeddings with one fixed model.
5. Build `LSM_VECTOR` and `FULL_TEXT` indexes.
6. Implement dense-only and BM25-only retrieval.
7. Build a small, judged query set and record baseline metrics.

**Exit criterion:** retrieval and evaluation run deterministically from a versioned corpus snapshot.

### Phase 2 — Native fusion

1. Implement dense + BM25 `vector.fuse()` with RRF.
2. Compare candidate depths and fused limits.
3. Add DBSF and LINEAR.
4. Add learned sparse embeddings and `LSM_SPARSE_VECTOR` if model output is available.
5. Record per-leg ranks and scores.

**Exit criterion:** reproducible dense/BM25/sparse comparisons with identical query and corpus inputs.

### Phase 3 — Graph retrieval

1. Validate entity resolution quality.
2. Create typed, provenance-bearing edges.
3. Implement seed-based bounded traversal.
4. Map graph discoveries back to evidence chunks.
5. Add graph candidate ranking or a graph feature vector.
6. Compare post-fusion graph expansion with parallel graph retrieval.

**Exit criterion:** graph evidence is bounded, auditable, and traceable to source chunks.

### Phase 4 — Reranking and RAG

1. Apply a cross-encoder to a fixed candidate pool.
2. Build source-aware context selection.
3. Generate answers with citations to source chunks.
4. Evaluate correctness, completeness, faithfulness, and citation quality.

**Exit criterion:** retrieval and answer-level metrics can be reported separately.

### Phase 5 — Analysis and thesis reporting

1. Run the complete ablation matrix.
2. Compute confidence intervals and paired comparisons.
3. Analyse query categories and failure cases.
4. Report retrieval quality against latency/storage/ingestion costs.
5. Document limitations, threats to validity, and reproducibility artifacts.

---

## 13. Risks, limitations, and methodological cautions

### 13.1 Graph quality can dominate retrieval quality

If entity resolution merges unrelated entities, graph traversal propagates incorrect relationships. If extraction misses relationships, graph retrieval cannot recover them. Measure graph precision/recall on a manually reviewed subset.

### 13.2 Graph degree and traversal explosion

A few high-degree entities can dominate expansion. Control with:
- edge-type allowlists;
- depth limits;
- maximum seed count;
- per-seed expansion limits;
- edge confidence thresholds;
- hubness penalties;
- source-document deduplication.

### 13.3 Leakage between corpus and evaluation

If query generation, graph extraction, or reranker tuning uses the test set, evaluation becomes optimistic. Separate development, validation, and test queries. If the corpus contains near-duplicates, split by source document or topic where feasible.

### 13.4 Score comparability

Cosine similarity, BM25, sparse dot product, graph confidence, and cross-encoder logits are not inherently calibrated to the same scale. Prefer RRF initially. Use score fusion only with a documented calibration procedure and held-out validation.

### 13.5 Retrieval diversity versus relevance

Source-level grouping can improve diversity but may discard multiple genuinely useful passages from one source. Measure both relevance and source coverage.

### 13.6 Context budget

A larger candidate set does not guarantee a better LLM answer. Long contexts can introduce redundancy and distractors. Measure final context size, redundancy, and answer quality under a fixed token budget.

### 13.7 Native graph expansion constraints

ArcadeDB's native hybrid graph expansion is convenient, but it has API-specific constraints. The documented gRPC interface requires RRF for the expansion leg and bounds graph depth. For more elaborate graph scoring or multi-stage traversal, a separately controlled graph candidate pipeline may be more suitable.

### 13.8 Version drift

ArcadeDB is actively evolving. Record the exact server release and verify query syntax against the running instance. Do not mix benchmark results from different versions without treating version as an experimental factor.

---

## 14. Recommended initial configuration

For the first working implementation, use:

| Component | Recommendation |
|---|---|
| Chunk retrieval unit | `Chunk` vertex |
| Dense embedding | One fixed multilingual model appropriate to corpus languages |
| Dense index | `LSM_VECTOR`, cosine initially |
| Full text | `FULL_TEXT`, BM25 |
| Sparse retrieval | Add BGE-M3/SPLADE output only when ingestion can reliably persist the exact sparse representation |
| Initial fusion | `vector.fuse()` with RRF |
| Per-leg candidates | 100 |
| Fused candidates | 50 |
| Graph expansion | Seed from top fused chunks/entities; depth 1–2; explicit edge allowlist |
| Graph evidence | Map back to supporting chunks and retain paths |
| Reranking | Cross-encoder over fused/expanded candidates |
| Final context | 5–15 evidence chunks, subject to token budget |
| Evaluation | Recall@K, nDCG@K, MRR, p50/p95 latency, answer faithfulness and citation quality |

These are starting parameters for a controlled experiment, not claimed optimal values.

**Most important recommendation:** establish dense-only, BM25-only, and dense+BM25 RRF baselines before adding graph complexity. Then measure the graph contribution with an ablation that changes only graph retrieval. This gives the thesis a defensible causal comparison instead of a single complex system whose gains cannot be attributed to any one component.

---

## 15. Further research directions

1. **Evidence-aware graph fusion:** rank graph-derived chunks using the quality and diversity of source evidence, rather than path length alone.
2. **Relation-aware query routing:** infer whether a question requires entity resolution, relation traversal, or ordinary semantic retrieval.
3. **Uncertainty-aware graph expansion:** expand only when entity resolution or retrieval confidence justifies it.
4. **Temporal graph retrieval:** preserve validity intervals and source timestamps so that obsolete relationships do not outrank current evidence.
5. **Contradiction-aware retrieval:** retrieve and present conflicting claims with their provenance instead of collapsing them.
6. **Adaptive candidate budgets:** allocate more candidates to retrieval legs that historically contribute unique relevant evidence for a query class.
7. **Graph construction quality versus retrieval quality:** independently vary graph extraction thresholds and measure the effect on downstream retrieval.

---

## 16. Primary technical references

The following are the primary sources used for the ArcadeDB-specific implementation details. Documentation and release notes should be rechecked against the exact server version used in the thesis.

1. **ArcadeDB 26.11.1 release notes** — version-specific Cypher 25 vector `SEARCH`, index DDL, filtered search, and performance/upgrade notes.  
   https://arcadedb.com/blog/arcadedb-26-11-1/

2. **ArcadeDB changelog** — current release and chronological release history.  
   https://arcadedb.com/changelog.html

3. **Vector Search** — dense and sparse index behaviour, `vector.neighbors`, `vector.sparseNeighbors`, `vector.fuse`, fusion strategies, reranking, grouping, and filtering.  
   https://docs.arcadedb.com/arcadedb/concepts/vector-search

4. **Full-Text Index and BM25 Ranking** — Lucene index setup, BM25, `SEARCH_INDEX`, `SEARCH_FIELDS`, and Cypher full-text procedures.  
   https://docs.arcadedb.com/arcadedb/how-to/data-modeling/full-text-index

5. **Indexes** — index types and use cases.  
   https://docs.arcadedb.com/arcadedb/concepts/indexes

6. **SQL TRAVERSE** — traversal syntax, depth, strategy, filtering, and path variables.  
   https://docs.arcadedb.com/arcadedb/reference/sql/sql-traverse

7. **gRPC API** — vector, hybrid, and full-text search request semantics, including optional graph expansion.  
   https://docs.arcadedb.com/arcadedb/reference/grpc-api

8. **Studio Query Panel** — native vector, hybrid, and full-text search UI.  
   https://docs.arcadedb.com/arcadedb/tools/studio/query

9. **ArcadeDB repository and releases** — source, tags, and release artifacts.  
   https://github.com/ArcadeData/arcadedb/releases

---

## 17. Conclusion

ArcadeDB 26.11.1 provides a credible native foundation for a multi-retriever RAG system: dense vectors, learned sparse vectors, BM25 full text, server-side fusion, vector reranking, graph traversal, and Cypher vector search are available within one multi-model database.

For a master's thesis, the strongest contribution is unlikely to be merely demonstrating that these features can be combined. The research value lies in **measuring when graph-derived evidence improves retrieval, how that improvement depends on query type and graph quality, and what latency and storage costs it introduces**.

A rigorous study should therefore progress from simple baselines to controlled ablations, preserve provenance for every graph-derived relationship, and evaluate both retrieval quality and downstream answer grounding. The recommended initial architecture is dense + BM25 RRF, followed by bounded, provenance-aware graph expansion and cross-encoder reranking.
