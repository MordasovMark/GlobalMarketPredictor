/**
 * Static metadata for the 20 dashboard stocks. Prices, ratios and ratings are
 * NOT stored here — they come live from the API (/api/universe, /api/quotes).
 */
export const TOP_US_STOCKS = [
  { ticker: 'AAPL', name: 'Apple Inc.', domain: 'apple.com' },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', domain: 'nvidia.com' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', domain: 'microsoft.com' },
  { ticker: 'TSLA', name: 'Tesla Inc.', domain: 'tesla.com' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', domain: 'amazon.com' },
  { ticker: 'META', name: 'Meta Platforms', domain: 'meta.com' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', domain: 'google.com' },
  { ticker: 'BRK.B', name: 'Berkshire Hathaway', domain: 'berkshirehathaway.com' },
  { ticker: 'V', name: 'Visa Inc.', domain: 'visa.com' },
  { ticker: 'UNH', name: 'UnitedHealth Group', domain: 'unitedhealthgroup.com' },
  { ticker: 'JPM', name: 'JPMorgan Chase', domain: 'jpmorganchase.com' },
  { ticker: 'JNJ', name: 'Johnson & Johnson', domain: 'jnj.com' },
  { ticker: 'WMT', name: 'Walmart Inc.', domain: 'walmart.com' },
  { ticker: 'XOM', name: 'Exxon Mobil Corp.', domain: 'exxonmobil.com' },
  { ticker: 'MA', name: 'Mastercard Inc.', domain: 'mastercard.com' },
  { ticker: 'AVGO', name: 'Broadcom Inc.', domain: 'broadcom.com' },
  { ticker: 'PG', name: 'Procter & Gamble', domain: 'pg.com' },
  { ticker: 'ORCL', name: 'Oracle Corp.', domain: 'oracle.com' },
  { ticker: 'COST', name: 'Costco Wholesale', domain: 'costco.com' },
  { ticker: 'HD', name: 'Home Depot', domain: 'homedepot.com' },
];

export function getStockMeta(ticker) {
  const u = String(ticker || '').toUpperCase();
  return TOP_US_STOCKS.find((s) => s.ticker === u);
}

// Fallback company descriptions when the live profile is unavailable.
export const STOCK_ABOUT = {
  AAPL:  "Apple Inc. designs, manufactures and markets consumer electronics, software and services. Its product ecosystem spans the iPhone, Mac, iPad, Apple Watch and AirPods, bolstered by a rapidly expanding suite of subscription services — iCloud, Apple TV+, Apple Arcade and Apple Pay — that drive high-margin recurring revenue.",
  NVDA:  "NVIDIA Corporation is the world leader in accelerated computing. Its GPU architecture powers modern AI training and inference workloads across hyperscale data centers, autonomous vehicles and robotics. The CUDA software platform creates deep developer lock-in, while partnerships with every major cloud provider cement its role as the backbone of the global AI infrastructure buildout.",
  MSFT:  "Microsoft Corporation delivers cloud computing, productivity software and enterprise services to individuals and organizations worldwide. Azure is the second-largest cloud platform globally, Microsoft 365 commands the enterprise productivity market, and the company's early OpenAI partnership has positioned Copilot as the leading enterprise AI assistant.",
  TSLA:  "Tesla Inc. designs and manufactures electric vehicles, energy storage systems and solar products. Beyond hardware, Tesla's software-defined vehicle platform and Full Self-Driving subscription generate recurring revenue. The company's Gigafactory network provides scale advantages, though margin pressure from aggressive pricing cycles remains a near-term watch item.",
  AMZN:  "Amazon.com is the world's largest e-commerce and cloud computing company. AWS — its cloud division — contributes the majority of operating income despite representing a fraction of revenue. Prime membership, third-party seller services, digital advertising and an expanding healthcare vertical (One Medical, Amazon Pharmacy) provide multiple durable growth vectors.",
  META:  "Meta Platforms operates the world's largest social media ecosystem — Facebook, Instagram, WhatsApp and Threads — reaching over 3 billion daily active users. Advertising monetization drives near-term profits while heavy investment in Reality Labs and AI infrastructure positions Meta for the next computing platform. Improving cost discipline has reignited margin expansion.",
  GOOGL: "Alphabet Inc. is the parent of Google, which commands roughly 90% of global search and powers the digital advertising duopoly alongside Meta. Google Cloud is the fastest-growing of the hyperscaler trio. YouTube, Waymo, DeepMind and a prolific venture portfolio offer optionality far beyond core search, while AI integration across all products defends its moat.",
  'BRK.B': "Berkshire Hathaway is Warren Buffett's diversified conglomerate, holding wholly-owned businesses spanning insurance (GEICO, Gen Re), railroads (BNSF), utilities and manufacturing alongside a massive publicly-traded equity portfolio anchored by Apple, Bank of America and Coca-Cola. The unmatched insurance float provides low-cost capital to fund acquisitions indefinitely.",
  V:     "Visa Inc. operates the world's largest retail electronic payments network, enabling transactions in over 200 countries across 4 billion credentials. The asset-light model collects a small fee on every transaction without taking credit risk, generating extraordinary free cash flow. Digital payment volume growth and tap-to-pay adoption in emerging markets are multi-decade secular tailwinds.",
  UNH:   "UnitedHealth Group is the largest US health insurer and one of the most vertically integrated healthcare companies globally. Its Optum segment — combining pharmacy benefit management, healthcare services and analytics — generates diversified revenue streams that reduce dependence on traditional insurance margins. Scale advantages and data assets are formidable competitive moats.",
  JPM:   "JPMorgan Chase is the largest US bank by assets, operating a best-in-class franchise across consumer banking, commercial lending, investment banking and asset management. Industry-leading technology investment, a fortress balance sheet and disciplined underwriting have produced consistent returns through multiple credit cycles. CEO Jamie Dimon's stewardship commands premium market confidence.",
  JNJ:   "Johnson & Johnson is a global pharmaceutical and MedTech leader following its consumer health spinoff (Kenvue). The innovative medicine segment focuses on oncology, immunology and neuroscience, while MedTech covers surgical robotics, orthopaedics and vision. A deep pipeline, AAA credit rating and 60+ consecutive years of dividend increases define its blue-chip status.",
  WMT:   "Walmart Inc. is the world's largest retailer by revenue, operating nearly 11,000 stores across 24 countries under multiple banners. Its membership program (Walmart+), first-party advertising business and fast-growing e-commerce operations are structurally shifting the margin profile higher. Supply chain scale provides unmatched pricing power in any economic environment.",
  XOM:   "Exxon Mobil Corporation is one of the world's largest publicly traded oil and gas companies, with integrated operations spanning upstream exploration, refining, chemicals and lubricants. The $60B Pioneer Natural Resources acquisition meaningfully expanded Permian Basin output. Capital discipline, a strong balance sheet and a growing low-carbon portfolio in carbon capture and hydrogen round out the investment thesis.",
  MA:    "Mastercard Incorporated operates a global payments technology network connecting 3+ billion cardholders with 100+ million merchants in over 210 countries. Like Visa, its fee-based model carries zero credit risk while capturing value from every swipe, tap and click. Mastercard's Vocalink and real-time payments capabilities are extending relevance well beyond card rails.",
  AVGO:  "Broadcom Inc. is a global semiconductor and infrastructure software leader. Its custom ASIC business designs AI networking chips for Google, Meta and ByteDance, making it a key beneficiary of hyperscaler AI capex. The $69B VMware acquisition adds a large, recurring software revenue stream, transforming Broadcom into a hybrid chip-and-software franchise with expanding margins.",
  PG:    "Procter & Gamble Co. is a global consumer goods company with a portfolio of 65+ brands spanning fabric care, home care, baby care, feminine care and grooming — including Tide, Pampers, Gillette and Oral-B. Pricing power, category leadership in premium segments and a relentless focus on productivity savings sustain mid-single-digit EPS growth through commodity and consumer cycles.",
  ORCL:  "Oracle Corporation provides database technology, cloud infrastructure (OCI) and enterprise applications to organizations worldwide. OCI's rapid growth and competitive pricing against AWS and Azure are attracting workload migrations, while the AI demand for Oracle's GPU clusters has driven a backlog surge. However, legacy on-premise licensing headwinds and elevated debt from cloud investment warrant monitoring.",
  COST:  "Costco Wholesale Corporation operates members-only warehouse clubs offering a curated selection of merchandise at thin margins, monetizing primarily through annual membership fees. This model drives exceptional member loyalty (>90% renewal rates), high inventory turns and below-market pricing that is structurally difficult to replicate. E-commerce and international expansion extend the long-term runway.",
  HD:    "The Home Depot is the world's largest home improvement retailer, serving both professional contractors (Pro) and do-it-yourself consumers. The Pro segment — roughly half of revenue — drives higher ticket sizes and repeat purchasing. The SRS Distribution acquisition deepens Pro penetration in specialty trades. Near-term results face headwind from a soft housing turnover cycle.",
};
