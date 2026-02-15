# Lost Deal Postmortem: GlobalBank Corp

**Deal lost: November 2025 | Deal ID: GB-2025-0112**

## Customer Profile

GlobalBank Corp is a top-20 US bank headquartered in Charlotte, NC. Annual revenue: $18.5B. Total employees: 45,000. Their analytics and risk division had 2,000 licensed analytics users across compliance, risk, trading, and retail banking. They were replacing an aging MicroStrategy deployment.

## Competitive Landscape

GlobalBank evaluated four vendors:

- **DataFlow Inc** (winner)
- **NovaSight Analytics** (second place)
- **Acme Analytics** (eliminated in round 1)
- **InsightCorp** (eliminated in round 1)

## Why We Lost

### 1. Compliance Gap

DataFlow's compliance portfolio was significantly more mature for regulated financial services:

- DataFlow had FedRAMP Moderate authorization (granted 2023)
- DataFlow had completed OCC (Office of the Comptroller of the Currency) regulatory audits
- DataFlow had 12 existing banking customers as references
- NovaSight had SOC2 Type II and HIPAA-ready, but no FedRAMP and no banking-specific compliance history

The CISO's office scored DataFlow 95/100 on security and NovaSight 72/100. This was the single largest factor in the decision.

### 2. Risk Assessment

GlobalBank's vendor risk management team flagged NovaSight as "medium risk" due to:

- Company size (280 employees vs DataFlow's 1,400)
- Time in market (founded 2018 vs DataFlow's 2016, and DataFlow is public)
- No existing financial services customers with 1,000+ seats
- No FedRAMP authorization

DataFlow was rated "low risk" across all categories.

## What We Did Right

### Technical Evaluation Win

We scored highest in the technical evaluation phase:

- **Technical demo score**: NovaSight 92/100, DataFlow 78/100
- **POC success rate**: NovaSight completed all 15 test scenarios, DataFlow completed 11 of 15
- **API evaluation**: NovaSight rated "excellent," DataFlow rated "adequate"
- **User feedback from pilot group**: NovaSight NPS 76, DataFlow NPS 41

### Competitive Pricing

Our pricing was significantly more competitive:

- **NovaSight**: 2,000 seats at $55/seat/month = $1,320K/year, with volume discount (15%) = $1,122K/year
- **DataFlow**: 2,000 seats at $80/seat/month = $1,920K/year, with negotiated discount = $1,632K/year
- **Delta**: NovaSight was 31% cheaper on a per-seat basis

## Lessons Learned

1. **FedRAMP is table stakes for banking**: we cannot compete for large financial services deals without FedRAMP Moderate authorization
2. **Vendor risk scoring matters as much as product quality**: GlobalBank's risk team had effective veto power regardless of technical evaluation results
3. **Banking references are essential**: we need 2-3 marquee financial services customers to establish credibility in the vertical
4. **CISO engagement must happen early**: we engaged GlobalBank's CISO in week 6; DataFlow engaged their CISO in week 1

## Action Items (Status as of January 2026)

- **FedRAMP Moderate authorization**: application submitted October 2025, estimated completion Q3 2026
- **Financial services case studies**: working with 2 current fintech customers (LendFast, PayStream) to develop banking-adjacent case studies
- **Banking compliance checklist**: created a pre-qualification checklist for banking deals to identify compliance gaps before investing in pursuit
- **CISO engagement playbook**: new sales process requires CISO-level meeting within first 2 weeks of any regulated industry deal
