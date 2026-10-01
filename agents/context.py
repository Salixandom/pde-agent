"""Shared runtime context wiring the FEM solver, knowledge graph,
and run store together for the agent tools to use."""
from __future__ import annotations
from dataclasses import dataclass

from knowledge_graph.kg_store import KnowledgeGraph, seed_default_materials
from database.run_store import RunStore


@dataclass
class AppContext:
    kg: KnowledgeGraph
    db: RunStore

    @classmethod
    def build(cls, db_path: str = "runs.db") -> "AppContext":
        kg = KnowledgeGraph()
        seed_default_materials(kg)
        db = RunStore(db_path)
        # Restore warm-start index from any completed runs already in SQLite,
        # so retrieval history survives application restarts.
        for run in db.list_recent(limit=10000):
            if run["status"] == "completed":
                rid = run["run_id"]
                kg._run_texts[rid] = run["description"] or ""
                import json as _json
                kg.g.add_node(
                    f"run:{rid}", kind="Run",
                    description=run["description"] or "",
                    config=run["config_json"] or "{}",
                    timestamp=run["created_at"],
                )
        if kg._run_texts:
            kg._rebuild_index()
        return cls(kg=kg, db=db)
