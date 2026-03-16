from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from telegram_sources import normalize_tg_username


@dataclass(frozen=True, slots=True)
class GuideSourceSpec:
    username: str
    profile_slug: str
    profile_kind: str
    display_name: str
    marketing_name: str | None
    source_kind: str
    trust_level: str
    base_region: str = "Калининградская область"
    priority_weight: float = 1.0
    flags: dict[str, Any] | None = None
    notes: str | None = None


def canonical_guide_sources() -> tuple[GuideSourceSpec, ...]:
    raw: list[GuideSourceSpec] = [
        GuideSourceSpec(
            username="tanja_from_koenigsberg",
            profile_slug="tatyana-udovenko",
            profile_kind="person",
            display_name="Татьяна Удовенко",
            marketing_name="Татьяна Удовенко",
            source_kind="guide_personal",
            trust_level="high",
            priority_weight=1.15,
            flags={"collaboration_heavy": True},
        ),
        GuideSourceSpec(
            username="gid_zelenogradsk",
            profile_slug="gid-zelenogradsk",
            profile_kind="person",
            display_name="Гид Зеленоградск",
            marketing_name="Гид Зеленоградск",
            source_kind="guide_personal",
            trust_level="high",
            flags={},
        ),
        GuideSourceSpec(
            username="katimartihobby",
            profile_slug="kati-marti",
            profile_kind="person",
            display_name="Катя Марти",
            marketing_name="Шаги Кати Марти",
            source_kind="guide_personal",
            trust_level="high",
            flags={"caption_heavy": True},
        ),
        GuideSourceSpec(
            username="amber_fringilla",
            profile_slug="amber-fringilla",
            profile_kind="person",
            display_name="Amber Fringilla",
            marketing_name="Amber Fringilla",
            source_kind="guide_personal",
            trust_level="high",
            priority_weight=1.05,
            flags={"collaboration_heavy": True},
        ),
        GuideSourceSpec(
            username="art_from_the_Baltic",
            profile_slug="art-from-the-baltic",
            profile_kind="project",
            display_name="Art from the Baltic",
            marketing_name="Art from the Baltic",
            source_kind="guide_project",
            trust_level="medium",
            priority_weight=1.0,
            flags={"mixed_topic": True},
            notes="Added from operator request on 2026-03-16; casebook expansion pending.",
        ),
        GuideSourceSpec(
            username="alev701",
            profile_slug="alev701",
            profile_kind="person",
            display_name="Алексей А.",
            marketing_name="alev701",
            source_kind="guide_personal",
            trust_level="medium",
            flags={},
        ),
        GuideSourceSpec(
            username="twometerguide",
            profile_slug="twometerguide",
            profile_kind="project",
            display_name="Двухметровый гид",
            marketing_name="Двухметровый гид",
            source_kind="guide_project",
            trust_level="medium",
            flags={"mixed_region": True},
        ),
        GuideSourceSpec(
            username="valeravezet",
            profile_slug="valera-vezet",
            profile_kind="project",
            display_name="Автобус Валера",
            marketing_name="Автобус Валера",
            source_kind="guide_project",
            trust_level="medium",
            flags={"promo_noise": True},
        ),
        GuideSourceSpec(
            username="ruin_keepers",
            profile_slug="ruin-keepers",
            profile_kind="organization",
            display_name="Хранители руин",
            marketing_name="Хранители руин",
            source_kind="organization_with_tours",
            trust_level="medium",
            priority_weight=1.05,
            flags={"organization": True},
        ),
        GuideSourceSpec(
            username="excursions_profitour",
            profile_slug="profitur",
            profile_kind="operator",
            display_name="Профи-тур",
            marketing_name="Профи-тур",
            source_kind="excursion_operator",
            trust_level="medium",
            flags={"operator": True, "on_request_heavy": True},
        ),
        GuideSourceSpec(
            username="vkaliningrade",
            profile_slug="v-kaliningrade",
            profile_kind="project",
            display_name="В Калининграде",
            marketing_name="В Калининграде",
            source_kind="aggregator",
            trust_level="medium",
            priority_weight=0.9,
            flags={"aggregator": True},
        ),
    ]
    out: list[GuideSourceSpec] = []
    for item in raw:
        username = normalize_tg_username(item.username)
        if not username:
            continue
        out.append(
            GuideSourceSpec(
                username=username,
                profile_slug=item.profile_slug,
                profile_kind=item.profile_kind,
                display_name=item.display_name,
                marketing_name=item.marketing_name,
                source_kind=item.source_kind,
                trust_level=item.trust_level,
                base_region=item.base_region,
                priority_weight=float(item.priority_weight or 1.0),
                flags=dict(item.flags or {}),
                notes=item.notes,
            )
        )
    out.sort(key=lambda item: item.username)
    return tuple(out)
