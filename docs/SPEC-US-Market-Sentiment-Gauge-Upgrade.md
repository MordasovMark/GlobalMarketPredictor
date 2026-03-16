# Product Spec: US Market Sentiment Gauge Upgrade

**Document owner:** BMAD Product Manager  
**Status:** Draft — Pending approval  
**Target:** US Market Dashboard — Sentiment card (frontend)  
**Last updated:** 2025-03-13  

---

## 1. Overview

### 1.1 Purpose
Upgrade the “US Market Sentiment” gauge section on the US Market Dashboard to a **premium, readable, and actionable** component. The current implementation has visual and UX limitations (label truncation, generic glow, basic animation, dense history table). This spec defines the requirements for a full upgrade without changing the underlying data contract (score 0–100, optional historical array).

### 1.2 Scope
- **In scope:** Visual design, typography, needle animation, trend data presentation, actionable insight copy, and card layout within the existing Sentiment card.
- **Out of scope:** Backend API changes, new data sources, and changes to other dashboard cards (Fear & Greed Timeline, tickers).

---

## 2. Requirements

### 2.1 Visual Overhaul

**Goal:** A premium UI with zone-based glow and fully readable category labels.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Dynamic glow** | Glow color and intensity must reflect the current sentiment zone. | • **Extreme Fear (0–24):** Deep red glow (e.g. `#7f1d1d` / `#991b1b` tint), stronger intensity.<br>• **Fear (25–44):** Orange/amber glow.<br>• **Neutral (45–55):** Subtle yellow/neutral glow.<br>• **Greed (56–75):** Green glow.<br>• **Extreme Greed (76–100):** Strong green glow (e.g. `#14532d` / `#166534` tint). |
| **Label readability** | Category labels must never be truncated (e.g. no “EXTRE…”). | • All five labels are fully visible: “Extreme Fear”, “Fear”, “Neutral”, “Greed”, “Extreme Greed”.<br>• Labels may use abbreviated forms only if the full form is available on hover/tooltip (e.g. “Ext. Fear” with tooltip “Extreme Fear”).<br>• Minimum font size and spacing so labels do not overlap or clip at 320px card width. |
| **Premium look** | Card and gauge should feel high-end (e.g. terminal / pro dashboard). | • Consistent use of card background (`#12131a`), subtle borders (`border-slate-800`), and refined shadows.<br>• Gauge arc and needle use a clear color scale (red → yellow → green) with smooth gradients.<br>• No visual clutter; hierarchy is clear (gauge → score → insight → trend). |

---

### 2.2 Animated Needle

**Goal:** A more realistic, spring-like needle motion.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Spring easing** | Needle movement must use spring physics, not a simple duration-based tween. | • Animation uses a spring configuration (e.g. stiffness + damping or equivalent spring-easing).<br>• Needle overshoots slightly past the target then settles (or damped oscillation), then comes to rest at the correct angle.<br>• No linear or ease-in-out-only animation for the needle. |
| **Correct mapping** | Needle angle must match score. | • Score 0 → needle at far left (Extreme Fear).<br>• Score 50 → needle at top (Neutral).<br>• Score 100 → needle at far right (Extreme Greed).<br>• Intermediate values interpolate linearly in angle. |
| **Performance** | Animation must remain smooth. | • No jank on 60 Hz displays; spring runs on GPU-friendly properties (e.g. `transform: rotate`). |

---

### 2.3 Contextual Data (Trend)

**Goal:** A small, scannable trend table for recent sentiment.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Time periods** | Show exactly three rows. | • **Current** — today’s (or latest) sentiment score.<br>• **1 Week Ago** — score from 7 days ago.<br>• **1 Month Ago** — score from ~30 days ago. |
| **Columns** | Each row shows period, score, and trend. | • **Period** (e.g. “Current”, “1 Week Ago”, “1 Month Ago”).<br>• **Score** (0–100, integer).<br>• **Trend vs current** (e.g. ↑ higher, ↓ lower, — same). Optional: small delta (e.g. +5, -12). |
| **Design** | Compact and consistent with card. | • Table is clearly separated (e.g. divider or spacing) from the score/insight block.<br>• Typography and colors match the rest of the card (muted headers, zone-colored score or trend).<br>• Fits within the card without horizontal scroll; readable on narrow widths. |

**Data:** If backend does not yet provide “1 Week Ago” and “1 Month Ago”, the UI may use placeholder/mock values with a note in the spec that these will be wired to real data later. The table structure and layout must support real data.

---

### 2.4 Actionable Insight

**Goal:** One clear, contextual sentence below the score that explains what this zone means for traders.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Placement** | Single sentence directly below the score (and zone label). | • Visible immediately after the main score/zone; above the trend table.<br>• Visually distinct (e.g. slightly smaller or muted) so it doesn’t compete with the score. |
| **Content** | Zone-specific, trader-oriented copy. | • **Extreme Fear:** e.g. “Market is oversold. Contrarian buying opportunity often identified.”<br>• **Fear:** e.g. “Risk-off sentiment. Consider defensive positioning or selective entries.”<br>• **Neutral:** e.g. “Sentiment balanced. Range-bound conditions typical.”<br>• **Greed:** e.g. “Risk-on. Stay selective with new entries; consider trimming extremes.”<br>• **Extreme Greed:** e.g. “Euphoria. Consider taking some risk off the table.”<br>• Copy is concise (one short sentence), non-promotional, and clearly tied to the current zone. |
| **Consistency** | Same tone and length across zones. | • All zones have exactly one sentence; style (e.g. sentence case, no exclamation marks) is consistent. |

---

### 2.5 Layout

**Goal:** Clean, balanced, premium card that fits the dashboard.

| Requirement | Description | Acceptance criteria |
|-------------|-------------|----------------------|
| **Structure** | Fixed content order. | 1. Card title: “US Market Sentiment”.<br>2. Gauge (arc + needle).<br>3. Category labels (full text, no truncation).<br>4. Score + zone name (prominent).<br>5. Actionable insight (one sentence).<br>6. Trend table (Current, 1 Week Ago, 1 Month Ago). |
| **Spacing** | Consistent and breathable. | • Use a single spacing scale (e.g. `gap-4` / `gap-6` or 8px/16px) between sections.<br>• No elements touching card edges without padding.<br>• Vertical rhythm: gauge → labels → score → insight → table. |
| **Responsiveness** | Works on small and large viewports. | • Card does not overflow; content reflows or scales so the gauge, labels, score, insight, and table remain usable (e.g. on 320px width and on 1920px).<br>• Needle and arc remain correctly aligned at all sizes. |

---

## 3. Non-Functional Notes

- **Accessibility:** Ensure score and zone are announced to screen readers; trend direction (↑/↓) has an accessible label (e.g. “Higher” / “Lower”).
- **Data:** Component continues to accept `value` (0–100) and optional `historical` (array of `{ period, value, label }`). Trend table shows exactly Current, 1 Week Ago, 1 Month Ago; if `historical` has different keys, mapping must be documented or normalized in the component.
- **No breaking changes:** Upgrade is confined to the Sentiment card UI and behavior; parent layout (e.g. grid column, dashboard container) is unchanged unless explicitly agreed.

---

## 4. Approval

| Role | Name | Date | Approved (Y/N) |
|------|------|------|----------------|
| Product Manager | | | |
| Design | | | |
| Engineering | | | |

**Approval checklist:**
- [ ] Visual overhaul (glow, labels, premium look) agreed
- [ ] Needle spring animation agreed
- [ ] Trend table (Current, 1W, 1M) agreed
- [ ] Actionable insight copy and placement agreed
- [ ] Layout and spacing agreed

---

*End of spec. No code changes until this document is approved.*
