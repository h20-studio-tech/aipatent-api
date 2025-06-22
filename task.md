Here is your **copy-paste ready plan** for handing off the embodiment labeling task, updated to use `ALD_GvHD_patent_segments.csv` as the official input:

---

## 🔧 Task: Embodiment Labeling of Patent Pages (`ALD_GvHD_patent_segments.csv`)

### 🎯 Goal

Annotate each page of the patent with structured **embodiment labels** based on semantic chunks extracted from the **Summary of Invention**, **Detailed Description**, and **Claims**.

---

### 📄 Input

Use this CSV file:

* `ALD_GvHD_patent_segments.csv`
  Columns:

  * `filename`: original PDF file name
  * `page_number`: page index
  * `section`: section title (e.g. `summary of invention`, `detailed description`, `claims`)
  * `text`: full page content (OCR)

---

### ✅ Output

Add **2 new columns** to the same CSV (without removing any rows):

* `chunks`: list of text chunks (each \~2–6 sentences)
* `embodiment_labels`: list of label objects (one per chunk), with structure:

```json
{
  "chunk": "text",
  "is_embodiment": true/false,
  "justification": "reasoning",
  "mapped_principles": ["P1", "P3"],
  "mapped_claims": [1, 3]
}
```

---

### 🧠 Embodiment Labeling Guidelines

For each chunk:

* Mark `is_embodiment: true` if the chunk describes a **concrete method, process, device, or variation** showing how the invention is realized in practice.
* Mark `is_embodiment: false` if it’s general context, explanation, or theory.

Label all chunks in rows where `section` is one of:

* `summary of invention`
* `detailed description`
* `claims`

---

### 📚 Invention Principles (Use to assign `mapped_principles`)

| ID | Description                                                                            |
| -- | -------------------------------------------------------------------------------------- |
| P1 | The invention treats Alcoholic Liver Disease (ALD) or Graft-versus-Host Disease (GvHD) |
| P2 | Uses antibodies derived from hyperimmunized egg product                                |
| P3 | Poultry are immunized using disease-related targets                                    |
| P4 | Antibodies are extracted and formulated into a composition                             |
| P5 | The antibody product is administered to a subject                                      |
| P6 | Includes excipients, stabilizers, or carriers                                          |
| P7 | Describes dose, frequency, or route of administration                                  |
| P8 | Product includes purified IgY or a whole egg/yolk mixture                              |

Also include the related claim numbers in `mapped_claims`.

---

### 📌 Constraints

* Each row = one page.
* **Do not merge or delete rows.**
* If a page has no embodiments or valid chunks, use empty arrays: `[]`
* Only modify rows from sections listed above

---

### 🛠 Tools Recommended

* Python + Pandas

* (Optional) [OpenAI + python-instructor](https://github.com/jxnl/instructor) for Pydantic-structured labeling output

---

### ✅ Deliverables

Return an updated version of `ALD_GvHD_patent_segments.csv` with the following **new columns**:

* `chunks`
* `embodiment_labels`

Ready for review and benchmarking.

---
