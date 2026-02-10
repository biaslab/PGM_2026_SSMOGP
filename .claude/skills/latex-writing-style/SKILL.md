---
name: latex-writing-style
description: Expert LaTeX writing assistant following academic conventions for mathematical papers. Use when writing or editing LaTeX files, managing citations, formatting equations, or structuring academic documents.
---

# LaTeX Paper Writing Style

Expert LaTeX writing assistant following academic conventions for mathematical papers.

## References and Citations

- Always use `\autoref{label}` or `\Cref{label}` for cross-references, never `\ref{label}` alone
- Use `\capsecref{sec:name}` for capitalized section references: "Section~X" (defined as `\newcommand{\capsecref}[1]{\hyperref[#1]{Section~\ref*{#1}}}`)
- Use `\refappx{appx:name}` for appendix references: "Appendix~X" (defined as `\newcommand{\refappx}[1]{\hyperref[#1]{Appendix~\ref*{#1}}}`)
- Use `\citet{key}` for textual citations: "As shown by \citet{smith2020}..."
- Use `\citep{key}` for parenthetical citations: "...has been established \citep{smith2020}."
- Never start a sentence with a citation in parentheses
- Use `\label{}` immediately after `\section`, `\subsection`, `\begin{equation}`, `\begin{figure}`, `\begin{table}`, etc.

## Label Naming Conventions

- Sections: `sec:name` (e.g., `sec:introduction`)
- Equations: `eq:name` (e.g., `eq:vfe`)
- Figures: `fig:name` (e.g., `fig:factor_graph`)
- Tables: `tab:name` (e.g., `tab:results`)
- Theorems: `thm:name` (e.g., `thm:main`)
- Lemmas: `lem:name` (e.g., `lem:helper`)
- Appendix sections: `appx:name` (e.g., `appx:proofs`)

## Math Notation

- Use `\bm{x}` for bold sequences of random variables, e.g., `\bm{x} = x_{k+1:T}`
- Individual time-indexed variables are not bolded: `x_t`, `u_t`, `y_t`
- Order variables by rate of change (fastest first): observations, states, controls, parameters
  - Example: `p(\bm{y}, \bm{x}, \bm{u}, \theta)` or `q(\bm{y}, \bm{x}, \theta | \bm{u})`
- Use `\hat{p}(\bm{x})` for preference/target distributions (hat for "desired")
- Use `\tilde{p}(\cdot)` for epistemic priors
- Use `\Ex{q}{...}` for expectations: `\mathbb{E}_{q}[...]`
- Use `\ent{q(x)}` for entropy: `\mathbb{H}[q(x)]`
- Use `\ent{q(x|y)}` for conditional entropy: `\mathbb{H}[q(x|y)]` (put the full conditional distribution inside)
- Use `\KL{q}{p}` for KL divergence: `\mathbb{D}_{\mathrm{KL}}[q \| p]`
- These commands use `\left`/`\right` by default; for manual sizing, pass an optional argument:
  - `\KL[big]{q}{p}` uses `\bigl`/`\bigr`
  - `\KL[Big]{q}{p}` uses `\Bigl`/`\Bigr`
  - `\KL[bigg]{q}{p}` uses `\biggl`/`\biggr`
  - `\KL[Bigg]{q}{p}` uses `\Biggl`/`\Biggr`
  - Same pattern applies to `\ent[Big]{...}` and `\Ex[Big]{q}{...}`
- Use `\dif` for differentials: `\int f(x) \dif x`
- Operators should use `\DeclareMathOperator`: `\DeclareMathOperator*{\argmin}{arg\,min}`
- Use `\text{}` for text within math mode, not `\mathrm{}` for words
- Subscripts that are words should use `\text{}`: `F_{\text{Bethe}}` not `F_{Bethe}`

## Equations

- Use `equation` environment for single important equations
- Use `align` for multi-line derivations (not `eqnarray`)
- Use `subequations` to group related equations with same number
- Use `split` inside `equation` to break long single equations
- Add `\label{}` to equations that will be referenced

### Equation Punctuation

Displayed equations are part of the sentence and must have appropriate punctuation:

- End with a **period** if the equation concludes a sentence
- End with a **comma** if the sentence continues after the equation (e.g., "where...")
- End with **no punctuation only** if followed by conditions on the same line (e.g., `\quad \text{if ...}`)
- In multi-line `align` environments, add punctuation to **each line** that ends a clause:
  - Intermediate lines typically end with `,` (comma)
  - The final line ends with `.` (period) or `,` if text follows
- Use `\,,` or `\,.` for proper spacing before punctuation in display math
- Example:
  ```latex
  \begin{align}
      F[q] &= \text{term}_1\,, \label{eq:first} \\
      &= \text{term}_2\,. \label{eq:second}
  \end{align}
  ```

## Figures and Tables

- Figures: caption goes below (`\caption{}` after `\includegraphics`)
- Tables: caption goes above (`\caption{}` before `\begin{tabular}`)
- Use `\centering` not `\begin{center}...\end{center}` inside floats
- Use `booktabs` package: `\toprule`, `\midrule`, `\bottomrule` (no vertical lines)
- Use `table*` or `figure*` for two-column spanning floats

## Writing Style

- Avoid "we" at the start of sentences when possible; vary sentence structure
- Define acronyms on first use: "Variational Free Energy (VFE)"
- Use "we" for actions taken in the paper, "the reader" or passive voice for general statements
- Prefer active voice when clarity permits
- Keep paragraphs focused on one idea
- Use `\emph{}` for emphasis and introducing terms, not bold
- Avoid the use of the em-dash as it reveals LLM-generated text

## Capitalization Conventions

### Named Quantities and Principles (Always Capitalize)
These are specific named technical terms in the Free Energy Principle literature:
- **Variational Free Energy** (VFE) - the specific functional $F[q]$
- **Expected Free Energy** (EFE) - the cost function $G(\bm{u})$
- **Free Energy Principle** (FEP) - the theoretical framework
- **Kullback-Leibler** (KL) divergence - proper name

### General Techniques and Methods (Always Lowercase)
These are general methodological terms, not specific named quantities:
- **variational inference** - the general technique/method
- **planning as inference** - the general approach/framework (unless starting a sentence)
- **active inference** - the general field/framework (same treatment as reinforcement learning)
- **reinforcement learning** - general field
- **optimal control** - general field
- **message passing** - general technique

### Context-Dependent Usage
- "Planning-as-Inference (PAI)" - capitalize when referring to the specific named framework introduced by Attias (2003)
- "Variational Free Energy minimization" - always capitalize because it refers to minimizing the specific VFE quantity
- "the Variational Free Energy functional" - capitalize when referring to the specific functional $F[q]$

### Examples
- ✓ "We minimize the Variational Free Energy functional $F[q]$"
- ✓ "The Expected Free Energy combines instrumental and epistemic value"
- ✓ "Planning is performed using variational inference"
- ✓ "The Planning-as-Inference (PAI) framework was introduced by Attias"
- ✓ "We use planning as inference to solve this problem"
- ✗ "We use Variational Inference to minimize the cost" (should be lowercase)
- ✗ "The expected free energy is defined..." (should be capitalized when referring to the EFE)

## Structure

- Each `\section` and `\subsection` should have a `\label`
- Use `%` comments to mark section boundaries and TODOs
- Keep one sentence per line for easier git diffs
- Use consistent indentation (2 or 4 spaces)

## Common Mistakes to Avoid

- Don't use `$$...$$` for display math; use `\[...\]` or `equation`
- Don't use `\def`; use `\newcommand`
- Don't load conflicting packages
- Don't hardcode spacing; let LaTeX handle it
- Don't use `\\ ` for paragraph breaks; use blank lines
- Don't mix American and British spelling
