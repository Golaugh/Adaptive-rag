

smart-planner/
├── README.md
├── requirements.txt                 # langgraph, langchain, fastapi, uvicorn, chromadb, httpx
├── .env.example                     # OPENAI_API_KEY / TAVILY_API_KEY (or Bing) etc.
├── data/
│   └── books/                       # your “planning” book files (pdf/txt/md)
├── var/
│   └── vectorstore/                 # persisted Chroma DB (gitignored)
├── src/
│   ├── main.py                      # FastAPI: POST /start, POST /resume (HITL), GET /plan/{id}
│   ├── graph.py                     # LangGraph: nodes + wiring (HITL + routing + synthesis)
│   ├── state.py                     # TypedDict for idea, summary, team, research, docs, plan, timeline
│   ├── services.py                  # LLM/embeddings, vectorstore (Chroma), web_search(), hybrid scoring
│   ├── ingest_books.py              # one-off loader/splitter/embed → var/vectorstore/
│   └── prompts/
│       ├── summarizer.md            # summarize idea for human review
│       ├── research.md              # web research prompt (risks, cautions)
│       ├── planner.md               # merge idea+findings into a plan
│       └── timeline.md              # convert plan → milestones with dates
└── tests/
    └── test_smoke.py                # tiny E2E: start → interrupt → resume → final plan


src/main.py

Minimal FastAPI with two endpoints:

POST /start → kicks off graph until HITL interrupt (returns session + draft summary).

POST /resume → accepts {session_id, updated_summary, approved} and continues to final plan/timeline.

This uses standard HITL via graph interrupt/resume instead of input() loops, which is the production pattern. (Router + interrupts are key to adaptive RAG.) 

src/graph.py

Nodes:

idea_summarize → drafts summary → HITL review (interrupt).

ask_team → collect “how many” + duties.

route_query → vectorstore vs. web decision (adaptive routing). 

retrieve_books → top-k from Chroma; optional simple hybrid fusion. 

grade_documents → binary “relevant?”; if “no”, branch to web_research. 

web_research → focused risks/cautions search; feeds back into context. 

synthesize_plan → merges idea + team + research + books.

build_timeline → turns plan into milestones/dates.

Optional self-checks you can keep off at first but the hooks are easy to add later: hallucination and answer graders. 

src/state.py

One TypedDict holding: idea, summary, team (size + duties), research_findings, book_chunks, plan_md, timeline.

src/services.py

llm() / embedder() factories

vectorstore() → Chroma persisted under var/vectorstore/ + a tiny hybrid_score(vec_score, kw_score)

If you later add a keyword/BM25 retriever, combine scores like:
final = 0.6*vector + 0.4*keyword (classic hybrid weighting). 

web_search(query) → Tavily or Bing wrapper.

src/ingest_books.py

Load/split all files in data/books/, embed, persist to Chroma. (Sets you up for local retrieval first; avoid web unless needed—adaptive mindset.) 

src/prompts/*.md

Keep prompts versioned and readable (summarize, research risks, plan, timeline). The generation chain can stay minimal: prompt → LLM → text. 

tests/test_smoke.py

One happy-path test: start → receive interrupt payload → resume with {approved: true} → check we got plan_md + timeline. You can grow tests later (router + graders).