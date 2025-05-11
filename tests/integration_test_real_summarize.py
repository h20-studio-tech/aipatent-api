import asyncio
from src.utils.ocr import summarize_embodiment, Embodiment

async def main():
    emb = Embodiment(
        text="The hyperimmunized egg product of claim 14, wherein the hyperimmunized egg product consists of purified or partially purified IgY antibody to Enterococcus faecalis and purified or partially purified IgY antibody to Enterococcus faecalis cytolysin toxin.",
        filename="test.pdf",
        page_number=1,
        section="detailed description",
        summary=""
    )
    result = await summarize_embodiment(emb)
    print("Summary:", result.summary)

if __name__ == "__main__":
    asyncio.run(main())
