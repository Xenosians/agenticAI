"""Versioned evidence boundary for owner-authorized retrieval adapters.

Adapters receive the authenticated owner from the host. They must authorize
before searching or resolving. A hit is evidence, never capability authority.
"""
from __future__ import annotations

import hashlib
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_name: Literal["owner-retrieval.v1"] = Field(default="owner-retrieval.v1", alias="schema")
    content: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=200)
    dataset_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    chunk: int = Field(ge=1, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    locator: str
    scope: Literal["owner_private"] = "owner_private"
    source_type: Literal["dataset"] = "dataset"
    trust: Literal["untrusted_retrieval"] = "untrusted_retrieval"

    @model_validator(mode="after")
    def verify_identity(self):
        if self.content_sha256 != hashlib.sha256(self.content.encode()).hexdigest():
            raise ValueError("Retrieved content hash does not match")
        if self.locator != citation_locator(self.dataset_id, self.chunk, self.sha256):
            raise ValueError("Retrieved locator does not match its source identity")
        return self


class OwnerRetrievalAdapter(Protocol):
    """Trusted adapter interface; browser-supplied owners are never accepted."""

    def search(self, owner: str, identity: str, query: str, limit: int = 5) -> list[dict]: ...

    def resolve(self, owner: str, identity: str, chunk: int, sha256: str) -> dict: ...


def citation_locator(identity: str, chunk: int, digest: str) -> str:
    return f"dataset:{identity}#chunk:{chunk}@{digest}"


def dataset_chunks(text: str) -> list[str]:
    return [text[index:index+2000] for index in range(0, len(text), 1800)]


def make_hit(resource: dict, chunk: int, content: str) -> dict:
    return RetrievalHit(content=content, source=resource["source"], license=resource["license"],
        dataset_id=resource["id"], chunk=chunk, sha256=resource["sha256"],
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        locator=citation_locator(resource["id"], chunk, resource["sha256"])).model_dump(by_alias=True)
