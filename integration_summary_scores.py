import sys
import os
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
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

def print_boxed(title, content, width=80):
    lines = content.splitlines() or ['']
    maxlen = min(width, max((len(line) for line in lines), default=0))
    border = f"+{'-' * (maxlen + 2)}+"
    print(border)
    print(f"| {title.ljust(maxlen)} |")
    print(border)
    for line in lines:
        print(f"| {line.ljust(maxlen)} |")
    print(border)

async def main():
    console = Console()
    console.rule("[bold cyan]Integration Summary Scores")
    for i, text in enumerate(texts, 1):
        emb = Embodiment(text=text, filename="test.pdf", page_number=i, section="detailed description", summary="")
        result = await summarize_embodiment(emb)
        src_len = len(text)
        sum_len = len(result.summary)
        score = sum_len / src_len if src_len > 0 else 0

        header = Text(f"Case {i} | Source: {src_len} | Summary: {sum_len} | Ratio: {score:.2f}", style="bold magenta")
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=2)
        table.add_column(justify="left", ratio=2)
        table.add_row(
            Panel(text, title="Input Text", border_style="cyan", box=box.ROUNDED),
            Panel(result.summary, title="Summary", border_style="green", box=box.ROUNDED),
        )
        console.print(header)
        console.print(table)
        console.rule()

if __name__ == "__main__":
    asyncio.run(main())
