import asyncio
from src.utils.ocr import summarize_embodiment, Embodiment

# List of test embodiment texts
texts = [
    "27. A method of preparing a hyperimmunized egg product comprising: hyperimmunizing an egg-producing animal with an antigen selected from the group consisting of Enterococcus faecalis, isolated Enterococcus faecalis cytolysin toxin, and Enterococcus faecium, and preparing a hyperimmunized egg product from one or more eggs produced by the animal.",
    "The hyperimmunized egg product of claim 14, wherein the hyperimmunized egg product consists of purified or partially purified IgY antibody to Enterococcus faecalis and purified or partially purified IgY antibody to Enterococcus faecalis cytolysin toxin.",
    "The hyperimmunized egg product of claim 14, wherein the hyperimmunized egg product is purified or partially purified IgY antibody to Enterococcus faecalis cytolysin toxin.",
    "In certain embodiments, administration of the hyperimmunized egg product to the subject recuces the level of Fnterococcus faecalis in the subject relative to a subject that ts not administered the hyperimmunized egg product.",
    "In certain aspects the disclosure relates to a method of preparing a hyperimmumized egg product comprising: hyperimmunizing an egg-producing animal with an antigen selected from the group consisting of Laterococcus faecalis, isolated Lnterococcus faecalis cytolysin toxin, and Knferococcus jaeciunt, and preparing a hyperimmunized egg product from one or more eges produced by the animal. In certain embodiments, the antigen is selected from the group consisting of isolated Anierecoccus faecalis cytolysin toxin, and fnterococcus faecium. In certain embodiments, the antigen is selected from the group consisting of Enterococcus faecalis and isolated Enierococcus faecalis cytolysin toxin. In certain embodiments, the antigen comprises Enterococcus faecalis and isolated Enterococcus faecalis cytolysin toxin. In certain embodiments, the egg-producing animal is a chicken.",
    "In some embodiments, the hyperimmunized egg product has higher levels of IgY antibodies to Aderococcus faeciem, interococcus jaecalis or Enterecoccus faecalis cytolysin toxin compared to a control egg product or an egg product from a chicken that has been immunized with favterococcus faecium, Enterococcus faecalis or Enterococcus jaecadis cvtolysin toxin using standard immunization techniques."
]

async def main():
    print("Running summarization and computing scores for test cases...\n")
    for i, text in enumerate(texts, 1):
        emb = Embodiment(text=text, filename="test.pdf", page_number=i, section="detailed description", summary="")
        result = await summarize_embodiment(emb)
        src_len = len(text)
        sum_len = len(result.summary)
        score = sum_len / src_len if src_len > 0 else 0
        print(f"Case {i}:")
        print(f"  Source length: {src_len}")
        print(f"  Summary length: {sum_len}")
        print(f"  Score (sum/src): {score:.2f}\n")
        print(f"  Summary: {result.summary}\n")

if __name__ == "__main__":
    asyncio.run(main())
