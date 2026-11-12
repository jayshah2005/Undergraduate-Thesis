---
name: literature-first
description: >-
  Before proposing or building a method, model, metric, preprocessing step or
  analysis, search for papers that already did it and start from their approach;
  cite them in code, READMEs and answers. Use whenever the user asks for
  something new to try, asks a "should we / would X work / what is standard"
  question, or asks to compare or choose between approaches.
---

# Literature first

Test things that already work. Before inventing anything, find what published work did for the same problem and build from that.

## When the user asks for something

1. **Search first.** Look for papers on the same task and data type before designing:
   - hyperspectral leaf or plant disease, grapevine viruses (GLRaV-3, GRBV);
   - small datasets;
   - Specim IQ or 400–1000 nm data.

   Then widen to general hyperspectral or vision work if nothing close exists. Use WebSearch, and read the abstract or method section, not just the title. The literature takeaways memory and `papers/` are a starting point, not the whole search.
2. **Prefer the published version.** Use the paper's architecture, layer sizes, loss and settings where they fit. Only invent a method when no paper covers the need, and say so ("no paper found; this repo's own design").
3. **Adapt openly.** Changes are fine when the data or this repo's rules force them, e.g. whole leaves instead of patches, masked batch norm, stride 4 for padding invariance, row-held-out folds. List each change and why next to the citation.
4. **Cite where it is used:**
   - in the class or function docstring: authors, year, title, venue (or arXiv id);
   - in the folder README;
   - in the answer to the user, with links.

   Never cite from memory alone. Confirm the title, authors and venue with a search before writing them down. If a paper can't be confirmed, say it is unconfirmed.
5. **Answer with evidence.** For a "would X work / is this standard" question, say what papers found, how they validated it, and whether that validation was as strict as ours (most grapevine papers use random splits, not held-out rows).

## Checklist

- [ ] Searched before designing; named the closest papers
- [ ] Used the published method where it fits; listed every adaptation and why
- [ ] Citations confirmed by search, in the docstring, README and answer
- [ ] Anything without a paper is labelled as the repo's own design
