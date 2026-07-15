# Hawaii proximal panel (`hawaii_proximal_panel.csv`)

Monthly Hawaii series, Jan 1991 – Dec 2020 (360 months), all as year-over-year
growth rates, with a `Mandatory Quarantine` treatment flag that switches on in
March 2020 (10 post-treatment months). Built for the proximal / negative-control
approach to isolating the border-closure effect from the pandemic, alongside the
Synthetic Historical Control panel it extends.

## Provenance

Two public sources, joined on month:

1. Tourism and labor outcomes — from `HawaiiData.xlsx`
   (`jgreathouse9/SynthWorld`, Paper 1): `Visitor Arrivals`, `Visitor Days`,
   `Occupancy`, `Mean Daily Rate`, `Revenue per Available Room`,
   `Total Leisure Emp`, `Unemp Rate`, `LFP`, `Accommodation Emp`,
   `Econ Activity Index`. These are the treated outcomes and are all
   *downstream* of the border closure.

2. Insulated-sector employment — from Hawaii DBEDT Monthly Economic Indicators
   (statewide employment by industry), converted here to YoY % growth:
   `NatRes_Constr_Emp`, `Wholesale_Emp`, `Financial_Emp`, `HealthCare_Emp`,
   `Government_Emp`. These are candidate negative controls: hit by the pandemic
   macro shock but *not* downstream of Hawaii's quarantine (they don't depend on
   whether visitors can arrive). In the pandemic trough these fell ~7–12% while
   the tourism series fell 50–99%.

## Roles

- Treated outcome: any of the tourism/labor columns (headline: `Visitor Days`).
- Negative controls (donors/proxies): the five `*_Emp` insulated sectors.
- Treatment: `Mandatory Quarantine` (1 from March 2020).

## Known limitation

The insulated sectors carry the pandemic's general-recession factor but not the
travel-demand-collapse factor (Hawaii tourism would have fallen from travel fear
even absent the mandate). Using them alone reconstructs only the macro component,
so the resulting proximal ATT is an upper bound (in magnitude) on the policy
effect. Closing that gap needs an external travel-demand donor — inbound
passengers to a comparable no-lockdown destination (e.g. Florida) — which is not
yet in this panel.
