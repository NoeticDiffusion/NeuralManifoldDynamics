#import "network_neuroscience_template.typ": *

#show: netn_template.with(
  title: "Style Guide for Network Neuroscience: Typst Template and Guidelines",
  short_title: "Network Neuroscience Style Guide",
  authors: (
    (name: "AI Assistant", affil: 1),
  ),
  affiliations: (
    "Cursor AI Development Team, Noetic Diffusion Project",
  ),
  corresponding_author: "AI Assistant, assistant@cursor.com",
  keywords: ("Network Neuroscience", "Typst", "Style Guide", "Template", "Scientific Publishing"),
  abstract: [
    This article serves as both a style guide and a practical template for writing articles intended for the journal Network Neuroscience using Typst. We summarize the formal requirements from the journal's guidelines and analyze three recently published articles to extract patterns for structure, figures, tables, and technical terms. Particular focus is placed on Typst-specific best practices, such as handling mathematical notation without LaTeX macros and efficient use of the template's features.
  ],
  author_summary: [
    Publishing in Network Neuroscience requires strict adherence to specific formal requirements, including word limits for the abstract and author summary, as well as a unique handling of technical terms. This guide simplifies the process by providing a ready-to-use Typst template that automatically handles formatting such as line spacing (double spacing), font (Times New Roman), and heading styles, while providing clear instructions for the content.
  ],
  author_contributions: [AI Assistant: Conceptualization, Writing – original draft.],
  funding_info: [No external funding was used for this project.],
  data_availability: [The template and style guide are available in the project's repository under `articles/aa_template/`.],
)

= Introduction
Network Neuroscience publishes innovative research on brain network organization. To facilitate the publication process, we have created this template in Typst. Typst offers a modern alternative to LaTeX with faster compilation and a more readable syntax, but requires some adjustment in how you write math and structure the document.

= Formal Requirements and Structure
Based on the journal's instructions (`article_info.md`) and the LaTeX template (`article_template.txt`), the following applies:

== Text Formatting
- *Font*: Times New Roman, 12pt.
- *Line Spacing*: Double spacing (in Typst: `leading: 1.8em`).
- *Margins*: Standard 1 inch (2.54 cm).

== Section Order
1. *Title and author information*: Including short title (max 70 characters).
2. *Abstract*: Max 200 words.
3. *Author Summary*: Max 125 words (required for Research articles).
4. *Introduction*: No subheadings allowed here.
5. *Results, Discussion, Methods*: Main sections with subheadings as needed.
6. *Acknowledgements*: Thanks to funders and contributors.
7. *References*: APA style required at the revision stage.
8. *Technical Terms*: Unique to Network Neuroscience. Approximately 10 terms with brief definitions.

= Writing in Typst (Best Practices)
According to `llm_typst_guide.md`, you should avoid LaTeX syntax directly in Typst.

== Math
Use Typst-native symbols instead of backslashes:
- Write `$alpha$` instead of `$\alpha$`.
- Write `$frac(a, b)$` instead of `$\frac{a}{b}$`.
- Write `$cal(M)$` for calligraphic letters.

Example of an equation:
$ rho^pi = frac(R I + op("E")_pi ([L, tau_L] | "post"), P + op("E")_pi ([L, tau_L] | "post") [tau_L]) $

== Technical Terms
In Network Neuroscience, the first occurrence of a *Technical Term* should be in bold. Definitions are listed at the end of the article. Use the template function `technical_terms()` for this:

```typst
#technical_terms((
  "Granger causality": "A procedure that estimates whether a time series can have a causal influence on another based on its past values.",
  "Neural mass": "A population of neurons that receive the same inputs and whose activity is strongly synchronized and coordinated.",
))
```

= Figures and Tables
Figures should be high resolution and referenced in the text. Tables should be formatted as Typst tables, not images.

#netn_box(title: "Example of a box (Boxed Text)")[
  Network Neuroscience often uses text boxes to highlight specific methods or results. In this template, you can use the function `#netn_box(title: "...")`.
]

= Conclusions
By using this template, you ensure that your article meets most formal requirements from the start. Remember to check word limits manually before submission.

#technical_terms((
  "Typst": "A modern, markup-based typesetting system designed to be more user-friendly than LaTeX.",
  "APA Style": "A referencing system created by the American Psychological Association, frequently used in neuroscience.",
))
