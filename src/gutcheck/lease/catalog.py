"""A catalog of things an agent could load: skills, MCP tools, subagents, docs, anything with a description."""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Item:
    id: str
    name: str
    description: str
    kind: str = "skill"                 # skill | mcp_tool | subagent | doc | ...
    token_cost: int = 0                 # context tokens the item costs when loaded
    payload: Dict[str, Any] = field(default_factory=dict)  # path, server, input_schema, ...

    @property
    def text(self) -> str:
        return "%s: %s" % (self.name.replace("__", " ").replace("_", " ").replace("-", " "), self.description)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def parse_frontmatter(text: str) -> Dict[str, str]:
    """Tiny YAML-frontmatter reader for SKILL.md / agent .md files (flat keys, folded values)."""
    m = _FM.match(text)
    if not m:
        return {}
    out: Dict[str, str] = {}
    key = None
    for line in m.group(1).splitlines():
        if re.match(r"^[A-Za-z0-9_-]+\s*:", line):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # a folded/literal block indicator (>, |, >-, |+ ...) means the value is on the next lines
            out[key] = "" if re.fullmatch(r"[>|][+-]?", val) else val.strip('"').strip("'")
        elif key and line.startswith((" ", "\t")):
            out[key] = (out[key] + " " + line.strip()).strip()
    return out


class Catalog:
    def __init__(self, items: Iterable[Item] = ()):
        self.items: List[Item] = []
        self._by_id: Dict[str, Item] = {}
        for it in items:
            self.add(it)

    def add(self, item: Item):
        if item.id in self._by_id:
            raise ValueError("duplicate catalog id %r" % item.id)
        if not item.token_cost:
            item.token_cost = estimate_tokens(item.text) + 10
        self.items.append(item)
        self._by_id[item.id] = item

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, id_: str) -> Item:
        return self._by_id[id_]

    def __contains__(self, id_: str) -> bool:
        return id_ in self._by_id

    @property
    def total_tokens(self) -> int:
        return sum(i.token_cost for i in self.items)

    # loaders -----------------------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "Catalog":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("items", [])
        return cls(Item(id=d["id"], name=d.get("name", d["id"]), description=d.get("description", ""),
                        kind=d.get("kind", "skill"), token_cost=int(d.get("token_cost", 0) or 0),
                        payload=d.get("payload", {})) for d in data)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(i) for i in self.items], f, indent=1, ensure_ascii=False)

    @classmethod
    def from_skill_dirs(cls, *roots: str, kind: str = "skill") -> "Catalog":
        """Every `<root>/<name>/SKILL.md` becomes an item (Claude Code / Agent Skills layout)."""
        cat = cls()
        for root in roots:
            for p in sorted(glob.glob(os.path.join(os.path.normpath(os.path.expanduser(root)), "*", "SKILL.md"))):
                with open(p, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                fm = parse_frontmatter(text)
                name = fm.get("name") or os.path.basename(os.path.dirname(p))
                desc = fm.get("description") or ""
                iid = "skill:" + name
                if iid in cat:
                    continue
                cat.add(Item(iid, name, desc, kind, estimate_tokens(name + desc) + 15,
                             {"path": p, "dir": os.path.dirname(p), "body_tokens": estimate_tokens(text)}))
        return cat

    @classmethod
    def from_agent_dirs(cls, *roots: str) -> "Catalog":
        """Every `<root>/*.md` with name/description frontmatter becomes a subagent item."""
        cat = cls()
        for root in roots:
            for p in sorted(glob.glob(os.path.join(os.path.normpath(os.path.expanduser(root)), "*.md"))):
                with open(p, encoding="utf-8", errors="replace") as f:
                    fm = parse_frontmatter(f.read())
                if not fm.get("name"):
                    continue
                iid = "agent:" + fm["name"]
                if iid in cat:
                    continue
                cat.add(Item(iid, fm["name"], fm.get("description", ""), "subagent",
                             estimate_tokens(fm["name"] + fm.get("description", "")) + 15, {"path": p}))
        return cat

    def merge(self, other: "Catalog") -> "Catalog":
        for it in other:
            if it.id not in self:
                self.add(it)
        return self
