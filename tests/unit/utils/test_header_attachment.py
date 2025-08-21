import types
import sys

import pytest

# Now we can safely import without the real 'instructor' package
from src.utils.ocr import add_headers_to_embodiments
from src.models.ocr_schemas import (
    DetailedDescriptionEmbodiment,
    HeaderDetectionPage,
)
# Stub heavy external deps so that importing src.utils.ocr works in a minimal test env
sys.modules.setdefault("instructor", types.ModuleType("instructor"))
sys.modules["instructor"].from_openai = lambda x: x  # type: ignore[attr-defined]



# Helper to build a DetailedDescriptionEmbodiment quickly

def make_emb(text: str, page: int, file: str = "file.pdf", header: str | None = None):
    return DetailedDescriptionEmbodiment(
        text=text,
        filename=file,
        page_number=page,
        section="detailed description",
        sub_category="product composition",
        header=header,
    )


# Helper to build HeaderDetectionPage

def make_header_page(
    header: str | None,
    has_header: bool,
    page: int,
    file: str = "file.pdf",
):
    return HeaderDetectionPage(
        filename=file,
        page_number=page,
        has_header=has_header,
        header=header,
        section="detailed description",
        text="",
        image=None,
    )


@pytest.mark.asyncio
async def test_direct_header_assignment():
    """Embodiments whose page has a detected header should receive it."""
    embs = [make_emb("E1", 1)]
    header_pages = [make_header_page("Formulation Examples", True, 1)]

    res = await add_headers_to_embodiments(embs, header_pages)
    assert res[0].header == "Formulation Examples"


@pytest.mark.asyncio
async def test_carry_forward_orphan():
    """A page without header inherits the previous page's header."""
    embs = [make_emb("E1", 1), make_emb("E2", 2)]
    header_pages = [
        make_header_page("Formulation Examples", True, 1),
        # page 2 has no header
    ]

    res = await add_headers_to_embodiments(embs, header_pages)
    assert res[0].header == "Formulation Examples"
    assert res[1].header == "Formulation Examples"  # carried forward


@pytest.mark.asyncio
async def test_first_page_without_header_stays_none():
    embs = [make_emb("E1", 1), make_emb("E2", 2)]
    header_pages = [
        make_header_page(None, False, 1),
        make_header_page("Intro", True, 2),
    ]

    res = await add_headers_to_embodiments(embs, header_pages)
    assert res[0].header is None  # nothing to carry yet
    assert res[1].header == "Intro"


@pytest.mark.asyncio
async def test_multiple_files_do_not_mix_headers():
    embs = [
        make_emb("E1", 1, "A.pdf"),
        make_emb("E2", 2, "B.pdf"),
        make_emb("E3", 3, "A.pdf"),
    ]
    header_pages = [
        make_header_page("Header A", True, 1, "A.pdf"),
        make_header_page("Header B", True, 2, "B.pdf"),
        # no header on A.pdf page 3
    ]

    res = await add_headers_to_embodiments(embs, header_pages)
    embs_by_text = {e.text: e for e in res}
    assert embs_by_text["E1"].header == "Header A"
    assert embs_by_text["E2"].header == "Header B"
    assert embs_by_text["E3"].header == "Header A"  # carried forward within A.pdf only
