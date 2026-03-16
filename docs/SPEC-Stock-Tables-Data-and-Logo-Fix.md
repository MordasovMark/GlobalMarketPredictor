# Technical Spec: Stock Tables — Data Integrity, Logos & Layout

**Document owner:** BMAD Product Manager  
**Priority:** High (UI/UX bug)  
**Target:** US Market Dashboard — Stock Tables section (`frontend/src/App.jsx`)  
**Last updated:** 2025-03-13  

---

## 1. Overview

### 1.1 Purpose
Fix high-priority UI/UX issues in the Stock Tables section: (1) “Top 20 Gainers” must display exactly 20 items and reflect correct ranking logic; (2) logos are missing across all three tables and must render reliably with a clear fallback; (3) table layout and scroll behavior must be consistent and professional (no messy per-table scrollbars, full-width alignment).

### 1.2 Scope
- **In scope:** Data source for the left table (20 items guarantee), logo column implementation and fallback, header/column alignment, scroll and width behavior for the entire Stock Tables section.
- **Out of scope:** Backend/API for live stock data, new table columns beyond Logo, and changes outside the Stock Tables block.

---

## 2. Requirements

### 2.1 Data Integrity — “Top 20 Gainers” (Left Table)

**Goal:** The left table MUST show exactly 20 rows and the title “Top 20 Gainers” must be accurate.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Source array length** | The data array used for the Stock Tables (e.g. `TOP_US_STOCKS` or equivalent) MUST contain at least 20 items. | • If the current constant has fewer than 20 entries, extend it to exactly 20 with real US tickers (e.g. AAPL, MSFT, NVDA, …).<br>• No placeholder or “empty row” hacks; each row must have valid ticker, name, price, changePercent, and other required fields. |
| **Rendering count** | The “Top 20 Gainers” table MUST render exactly 20 rows. | • The table body shows 20 `<tr>` elements.<br>• Sorting is applied on a copy of the array (e.g. by `changePercent` descending); the result is sliced or constrained to the first 20 items only if the source could ever exceed 20 (e.g. `.slice(0, 20)` after sort).<br>• If the source has fewer than 20 items (e.g. after a future API change), document the fallback: either pad with a defined fallback list up to 20, or show all available rows and do not claim “Top 20” when count &lt; 20 (e.g. title could be “Top Gainers” when count is variable). Preferred: ensure source always has 20 items. |
| **Ranking logic** | “Gainers” means sorted by price change (descending). | • Rows are ordered by `changePercent` descending (highest gain first).<br>• Ties may be broken by a secondary sort (e.g. ticker) for deterministic order. |

---

### 2.2 Logo Implementation

**Goal:** A dedicated “Logo” column on the far left of each table; logos load when possible and fallback is always visible and consistent.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Logo column** | Add a “Logo” column as the leftmost column in all three tables. | • **Table 1 (Top 20 Gainers):** Columns order: Logo, Symbol, Price, Change %.<br>• **Table 2 (AI Pro Predictions):** Columns order: Logo, Ticker, Signal, Confidence, Action.<br>• **Table 3 (Market Giants):** Columns order: Logo, Symbol, Name, Price.<br>• Header text: “Logo” (or empty header with appropriate `aria-label` for accessibility). |
| **Primary source** | Use Clearbit for company logos. | • URL pattern: `https://logo.clearbit.com/[domain].com`.<br>• Each row’s `logoUrl` (or equivalent) must use the correct company domain for the ticker (e.g. `apple.com` for AAPL, `microsoft.com` for MSFT).<br>• Tech note: Maintain a mapping or field so that [domain] is correct (e.g. BRK.B → berkshirehathaway.com, TSMC → tsmc.com). |
| **Fallback when load fails** | If the logo image fails to load (404, CORS, network, etc.), show a fallback. | • Fallback: a **stylized circular div** with fixed size **24×24px**.<br>• Background: **Slate-800** (or `#1e293b` / Tailwind `bg-slate-800`).<br>• Content: the **first letter of the ticker** (e.g. “A” for AAPL, “T” for TSLA), in a consistent style (e.g. centered, uppercase, neutral text color).<br>• No broken-image icon; the fallback must be the only visible content when the image fails.<br>• Implementation: use `onError` on the `<img>` to hide the image and show the circular fallback (e.g. via state, class toggle, or conditional render). Ensure the fallback is in the DOM and visible when the primary image fails. |
| **Size and position** | Logo cell content is consistent and aligned. | • Logo graphic (image or fallback circle): **24×24px** exactly.<br>• The Logo column is wide enough to contain the 24×24 asset with comfortable padding; the asset is **perfectly centered** within the Logo column (e.g. flex center or text-align center in a fixed-width column). |

---

### 2.3 Layout Alignment

**Goal:** Headers and row data align; Logo column is visually clear and centered.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Header–body alignment** | Table header and body columns align. | • Each `<th>` and corresponding `<td>` use the same alignment (e.g. text-left for Symbol/Ticker/Name/Action, text-right for Price/Change %/Confidence).<br>• Use consistent horizontal padding (e.g. same `px-*` or padding value) for the same column index in `<thead>` and `<tbody>`.<br>• Optional: use `table-fixed` and explicit column widths for the Logo column (e.g. width equivalent to 24px + padding) so alignment is stable across browsers. |
| **Logo column** | Logo column is dedicated and centered. | • Logo column: 24×24px content centered in the cell; no text in the Logo cell except the image or the single-letter fallback.<br>• Symbol/Ticker/Name (and other text columns) are in their own columns to the right of Logo. |

---

### 2.4 Professional Polish — Scroll and Width

**Goal:** No messy independent scrollbars; section uses full width of the container.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Scroll behavior** | Tables must not each show a separate scrollbar in a way that looks messy or inconsistent. | • **Option A (preferred):** Remove per-table scroll (e.g. remove `overflow-y-auto` and `max-h-[600px]` from each card). Let the Stock Tables section scroll with the page (one page scroll). If the section becomes long, the main content area scrolls; no scrollbars inside the table cards.<br>• **Option B:** If a max height is required, use a **single scrollable container** that wraps all three tables (e.g. one `overflow-y-auto max-h-[600px]` around the three-column grid), so there is only one scrollbar for the whole section.<br>• Do **not** keep three independent scrollable table bodies that each show a scrollbar when content overflows. |
| **Full width** | The Stock Tables section spans the full width of its container. | • The section uses the same horizontal space as the content above it (e.g. same `px-10` or container padding).<br>• The grid is `grid-cols-1 lg:grid-cols-3` (or equivalent) with `w-full` so on large screens the three columns share the full width; no unnecessary max-width that makes the section look narrow. |

---

## 3. Data Contract (Reference)

Each stock row must include at least:

- `ticker` (string)
- `name` (string)
- `price` (number)
- `changePercent` (number)
- `logoUrl` (string) — primary Clearbit URL: `https://logo.clearbit.com/[domain].com`
- `aiSignal` (string): e.g. "Strong Buy" | "Buy" | "Hold" | "Sell"
- `confidence` (number, optional for Table 2)
- `marketCap` (number, optional for Table 3 sorting)

For the Logo column, only `logoUrl` and `ticker` (for fallback letter) are required.

---

## 4. Implementation Notes (Non-Normative)

- **Clearbit:** Clearbit logo URLs may not load in all environments (CORS, referrer policy). The fallback (circular div with first letter) is therefore mandatory for a robust UI.
- **Accessibility:** Ensure the Logo column has an accessible name (e.g. “Logo” in header or `aria-label` on the cell) and that the fallback letter is not read redundantly if the image has empty `alt` (e.g. `alt=""` and fallback with `aria-hidden="true"` if appropriate).
- **Testing:** After implementation, verify with network throttling or blocked image domains that the fallback appears and that all three tables show the Logo column and aligned data.

---

## 5. Approval

| Role | Name | Date | Approved (Y/N) |
|------|------|------|----------------|
| Product Manager | | | |
| Engineering | | | |

**Checklist:**
- [ ] Data: 20 items guaranteed for Top 20 Gainers; ranking logic confirmed
- [ ] Logo: Column added; Clearbit URL; fallback circular 24×24 Slate-800 + first letter
- [ ] Layout: Headers and rows aligned; Logo 24×24 centered
- [ ] Polish: No per-table scrollbars (single scroll or page scroll); section full width

---

*End of spec. Implement only after approval.*
