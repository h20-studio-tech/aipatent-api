import pytest
import asyncio

from src.utils.ocr import (
    summarize_embodiment,
    summarize_embodiments,
    Embodiment,
    DetailedDescriptionEmbodiment,
    embodiment_summarization_prompt,
    client,
)

# Dummy prompt to replace the real one
class DummyPrompt:
    def compile(self, embodiment):
        return f"Compiled {embodiment}"

# Dummy response to mimic API behavior
class DummyResponse:
    def __init__(self, summary):
        self.summary = summary

@pytest.fixture(autouse=True)
def patch_client_and_prompt(monkeypatch):
    # Patch the global summary prompt
    monkeypatch.setattr(
        "src.utils.ocr.embodiment_summarization_prompt",
        DummyPrompt(),
    )
    # Patch client.chat.completions.create
    def dummy_create(model, messages, response_model):
        assert model == 'gpt-4.1'
        content = messages[0]['content']
        # Return a DummyResponse with predictable summary
        return DummyResponse(summary=f"Summary of {content}")

    # Replace the method
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        dummy_create,
    )

@pytest.mark.asyncio
async def test_summarize_embodiment_with_embodiment():
    emb = Embodiment(
        text='emb text',
        filename='file.pdf',
        page_number=1,
        section='detailed description',
        summary='',
    )
    result = await summarize_embodiment(emb)
    assert isinstance(result, Embodiment)
    assert result.summary == 'Summary of Compiled emb text'

@pytest.mark.asyncio
async def test_summarize_embodiments_list():
    emb1 = Embodiment(
        text='text1',
        filename='f1.pdf',
        page_number=1,
        section='summary of invention',
        summary='',
    )
    emb2 = DetailedDescriptionEmbodiment(
        text='text2',
        filename='f2.pdf',
        page_number=2,
        section='detailed description',
        sub_category='product composition',
        summary='',
    )
    results = await summarize_embodiments([emb1, emb2])
    assert len(results) == 2
    assert results[0].summary == 'Summary of Compiled text1'
    assert results[1].summary == 'Summary of Compiled text2'
