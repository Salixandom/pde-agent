"""
Lightweight knowledge-graph store: a networkx graph for structure
(Material / Run / KnownIssue nodes and their edges) plus a TF-IDF
+ cosine-similarity index for semantic "warm-start" retrieval.

This replaces the base paper's Neo4j + 768-d nomic-embed + HNSW
stack, which needs a running graph-database server and a
downloadable embedding model -- neither of which is available in
this sandbox. The architecture (graph of entities, embedding-based
similarity search over prior runs, warm-start injection before the
agent loop) is preserved; only the storage/embedding backend
differs. Swap in Neo4j + a real embedding model for a production
deployment without touching the agent code (agents/tools.py only
calls the methods below).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import time
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Material:
    name: str
    k: float           # W/(m K)
    rho: float          # kg/m^3
    cp: float           # J/(kg K)
    notes: str = ""


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._run_texts: dict[str, str] = {}   # run_id -> description text
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._run_matrix = None
        self._run_ids: list[str] = []

    # ---------------- materials ---------------- #
    def add_material(self, mat: Material):
        self.g.add_node(f"material:{mat.name}", kind="Material",
                         k=mat.k, rho=mat.rho, cp=mat.cp, notes=mat.notes)

    def get_material(self, name: str) -> Optional[Material]:
        node = f"material:{name}"
        if node not in self.g.nodes:
            return None
        d = self.g.nodes[node]
        return Material(name=name, k=d["k"], rho=d["rho"], cp=d["cp"], notes=d.get("notes", ""))

    def list_materials(self) -> list[str]:
        return [n.split(":", 1)[1] for n, d in self.g.nodes(data=True) if d.get("kind") == "Material"]

    # ---------------- known issues ---------------- #
    def add_known_issue(self, code: str, trigger_desc: str):
        self.g.add_node(f"issue:{code}", kind="KnownIssue", trigger=trigger_desc)

    # ---------------- run lineage + warm-start ---------------- #
    def record_run(self, run_id: str, description: str, config: dict, material: Optional[str] = None):
        self.g.add_node(f"run:{run_id}", kind="Run", description=description,
                         config=json.dumps(config), timestamp=time.time())
        if material:
            self.g.add_edge(f"run:{run_id}", f"material:{material}", kind="USES_MATERIAL")
        self._run_texts[run_id] = description
        self._rebuild_index()

    def _rebuild_index(self):
        if not self._run_texts:
            return
        try:
            run_ids = list(self._run_texts.keys())
            docs = [self._run_texts[r] for r in run_ids]
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            # Only commit if fit_transform succeeded
            self._run_ids = run_ids
            self._vectorizer = vectorizer
            self._run_matrix = matrix
        except Exception:
            pass  # Leave index in last good state; warm_start returns [] if unavailable

    def warm_start(self, task_description: str, top_k: int = 3) -> list[dict]:
        """Retrieve the top-k most similar past runs (cosine over
        TF-IDF vectors) to inject as few-shot context before the
        agent loop -- the paper's KG Smart warm-start pattern."""
        if not self._run_ids or self._vectorizer is None:
            return []
        qvec = self._vectorizer.transform([task_description])
        sims = cosine_similarity(qvec, self._run_matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        out = []
        for i in order:
            if sims[i] <= 0:
                continue
            rid = self._run_ids[i]
            node = self.g.nodes[f"run:{rid}"]
            out.append({
                "run_id": rid,
                "similarity": float(sims[i]),
                "description": node["description"],
                "config": json.loads(node["config"]),
            })
        return out

    def query(self, text: str) -> dict:
        """Lazy conditional retrieval: look up a material by
        (fuzzy) name match, mirroring the paper's
        `query_knowledge_graph` tool.
        Sorted by name length (longest first) so 'stainless_steel'
        matches before 'steel' when both substrings appear."""
        text_lower = text.lower().replace("_", " ")
        for name in sorted(self.list_materials(), key=len, reverse=True):
            normalized = name.lower().replace("_", " ")
            if normalized in text_lower:
                mat = self.get_material(name)
                return {"found": True, "material": mat.__dict__}
        return {"found": False, "material": None}


def seed_default_materials(kg: KnowledgeGraph):
    """Standard engineering materials, mirroring the base paper's
    curated library (Section 4.1)."""
    defaults = [
        Material("steel", k=50.0, rho=7850.0, cp=490.0, notes="AISI 1010 structural steel"),
        Material("copper", k=385.0, rho=8960.0, cp=385.0, notes="pure copper"),
        Material("aluminium", k=205.0, rho=2700.0, cp=900.0, notes="Al 6061"),
        Material("titanium", k=6.7, rho=4430.0, cp=526.0, notes="Ti-6Al-4V"),
        Material("stainless_steel", k=16.3, rho=8000.0, cp=500.0, notes="SS 304"),
    ]
    for m in defaults:
        kg.add_material(m)
    return kg


def seed_novel_materials(kg: KnowledgeGraph):
    """Fictional materials absent from any LLM's training data --
    the paper's novel-material stress test (Section 6.1)."""
    novel = [
        Material("novidium", k=73.0, rho=5420.0, cp=612.0, notes="fictional ceramic-metallic composite"),
        Material("cryonite", k=0.42, rho=1180.0, cp=1940.0, notes="fictional polymer-aerogel hybrid insulator"),
        Material("pyrathane", k=312.0, rho=3850.0, cp=278.0, notes="fictional refractory cermet, very high conductivity"),
    ]
    for m in novel:
        kg.add_material(m)
    return kg
