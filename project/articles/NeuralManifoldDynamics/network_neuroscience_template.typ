// Network Neuroscience Typst Template
// Based on official guidelines and recent publications.

#let netn_template(
  body,
  title: "",
  short_title: "",
  subtitle: none,
  authors: (),
  affiliations: (),
  corresponding_author: "",
  keywords: (),
  article_type: "Research",
  table_text_size: 9pt,
  table_leading: 1.2em,
  abstract: [],
  author_summary: none,
  author_contributions: none,
  funding_info: none,
  data_availability: none,
  competing_interests: "The authors have declared that no competing interests exist.",
) = {
  // Page setup
  set page(
    paper: "us-letter",
    margin: (x: 1in, y: 1in),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 9pt, style: "italic")
        if short_title != "" {
          short_title
        } else {
          title
        }
        h(1fr)
        counter(page).display()
      }
    }
  )

  // Text setup
  set text(
    font: "Times New Roman",
    size: 12pt,
  )

  // Tables
  // Network Neuroscience guidelines commonly use smaller table text.
  // This applies only to the table content (captions are typically outside `#table`).
  show table: it => {
    set text(size: table_text_size)
    set par(
      first-line-indent: 0pt,
      leading: table_leading,
    )
    it
  }

  // Paragraph setup
  set par(
    justify: true,
    first-line-indent: 0.5in,
    leading: 1.8em,
  )

  // Heading setup
  set heading(numbering: none)
  show heading: it => {
    set block(above: 1.5em, below: 1em)
    if it.level == 1 {
      set text(size: 14pt, weight: "bold")
      it
    } else if it.level == 2 {
      set text(size: 12pt, weight: "bold")
      it
    } else {
      set text(size: 12pt, weight: "bold", style: "italic")
      it
    }
  }

  // Title Page
  align(center)[
    #text(size: 10pt)[Article type: #article_type] \
    #v(1em)
    #text(size: 18pt, weight: "bold")[#title] \
    #if subtitle != none {
      text(size: 14pt, style: "italic")[#subtitle]
      linebreak()
    }
    #v(1em)
    #text(size: 10pt)[Short title: #short_title] \
    #v(1.5em)
    
    // Authors
    #let author_names = authors.map(a => {
      a.name + super(str(a.affil))
    }).join(", ")
    #text(size: 11pt)[#author_names] \
    #v(0.5em)
    
    // Affiliations
    #let affil_list = affiliations.enumerate().map(a => {
      super(str(a.at(0) + 1)) + " " + a.at(1)
    }).join("; ")
    #text(size: 9pt, style: "italic")[#affil_list] \
    #v(1em)
    
    // Corresponding author
    #text(size: 10pt)[Corresponding author: #corresponding_author] \
    #v(1em)
    
    // Keywords
    #text(size: 10pt)[Keywords: #keywords.join(", ")]
  ]

  v(2em)

  // Abstract
  if abstract != [] {
    heading(level: 1)[Abstract]
    par(first-line-indent: 0pt)[#abstract]
  }

  // Author Summary
  if author_summary != none {
    v(1.5em)
    heading(level: 1)[Author Summary]
    par(first-line-indent: 0pt)[#author_summary]
  }

  v(2em)

  // Reset paragraph indent for the first paragraph of sections
  show heading: it => {
    it
    par(first-line-indent: 0pt)[#text(size: 0pt, h(0pt))]
  }

  body

  // Author Contributions
  if author_contributions != none {
    heading(level: 1)[Author Contributions]
    author_contributions
  }

  // Funding Information
  if funding_info != none {
    heading(level: 1)[Funding Information]
    funding_info
  }

  // Data Availability
  if data_availability != none {
    heading(level: 1)[Data Availability Statement]
    data_availability
  }

  // Competing Interests
  if competing_interests != none {
    heading(level: 1)[Competing Interests]
    competing_interests
  }
}

// Helper for Technical Terms section
#let technical_terms(terms) = {
  heading(level: 1)[Technical Terms]
  for (term, definition) in terms {
    par(first-line-indent: 0pt)[
      *#term*: #definition
    ]
  }
}

// Helper for boxes (boxedtext/whitebox)
#let netn_box(body, title: none) = {
  block(
    width: 100%,
    fill: luma(245),
    inset: 1em,
    stroke: 0.5pt + luma(100),
    radius: 4pt,
  )[
    #if title != none {
      text(weight: "bold")[#title]
      v(0.5em)
    }
    #body
  ]
}
