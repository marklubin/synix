# MegaCorp Industries — Customer Requirements

**Deal ID: MC-2026-0142 | Last updated: February 2026**

## Company Background

MegaCorp Industries is a Fortune 500 manufacturing conglomerate headquartered in Chicago, IL. Annual revenue: $8.2B (FY2025). The company operates 34 manufacturing plants across North America and Europe. Total employees: 22,000. The Analytics & Business Intelligence division has 550 employees.

## Current State

MegaCorp currently uses a legacy Tableau Server deployment (version 2021.4) that has been in production since 2019. The Tableau environment supports 500 licensed analytics users and 50 power users who build and maintain dashboards. Annual Tableau spend is $280K, covering server licenses, viewer licenses, and Tableau-managed support.

## Pain Points with Current Solution

- Tableau Server is on-prem and requires significant IT overhead (2 FTEs dedicated to maintenance)
- No real-time data capabilities — dashboards refresh on a 4-hour schedule
- Power users report spending 40% of their time on data preparation rather than analysis
- No API access for programmatic dashboard creation or data extraction
- SSO integration is fragile and breaks after every Tableau upgrade
- No support for custom ML models — data science team uses separate Jupyter notebooks

## Requirements (Prioritized)

1. **Real-time data** (must-have): sub-minute data freshness for operational dashboards
2. **Custom reporting** (must-have): self-service report builder for business users without SQL knowledge
3. **API access** (must-have): full REST API for integration with internal systems (ERP, MES, CRM)
4. **SSO/SAML** (must-have): reliable integration with their Okta identity provider
5. **Custom ML models** (high priority): ability to deploy Python models trained by their data science team
6. **Migration tooling** (high priority): automated migration of existing Tableau dashboards
7. **Embedded analytics** (nice-to-have): embed dashboards in their customer portal
8. **On-prem data processing** (nice-to-have): sensitive manufacturing data stays in their data center

## Evaluation Criteria

MegaCorp has defined four weighted evaluation criteria:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Ease of migration | 30% | How quickly can we move from Tableau with minimal disruption |
| Time-to-value | 25% | How fast can business users start creating reports |
| Total cost of ownership | 25% | 3-year TCO including licenses, implementation, and training |
| Enterprise security | 20% | Compliance certifications, access controls, data residency |

## Decision Timeline

- **January 2026**: RFP distributed to 4 vendors (NovaSight, DataFlow, Acme, InsightCorp)
- **February 2026**: Vendor presentations and technical demos
- **March 2026**: Vendor selection and contract negotiation
- **April-May 2026**: Legal review and procurement (4-6 weeks)
- **July 2026**: Implementation start
- **October 2026**: Full production rollout target

## Budget

Approved budget range: $300K-500K annual for the analytics platform, inclusive of licensing, implementation services, and first-year support. Budget owner: David Kim, CFO.
