"""``pack.json`` schema (docs/ISL_DESIGN.md section 4.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

KINDS = ("experiment-registry", "isl-hybrid")


class LicenceRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    licence: str
    commercial: bool


class PackManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: int = Field(1, alias="schema")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    kind: str
    version: str | None = None
    sign_language: str
    standard: str = ""
    display_name: dict[str, str]
    output_languages: list[str] = Field(default_factory=lambda: ["en"])
    capabilities: list[str] = Field(default_factory=lambda: ["words"])
    profile: str = "research"
    feature_config: dict | None = None
    vocabulary: str | None = None
    lexicons: dict[str, str] = Field(default_factory=dict)
    components: dict = Field(default_factory=dict)
    results: str | None = None
    licences: list[LicenceRef] = Field(default_factory=list)
    note: str | None = None
