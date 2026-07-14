# `reunification_oecd23.csv`

An extended West German reunification panel for demonstrating donor-selection,
spillover-aware, and proximal synthetic-control methods against the critique of
Abadie, Diamond & Hainmueller (2015) in Francis (2025).

- 24 units (West Germany + the full pool of 23 OECD donors), 1960–2003.
- Columns: `country`, `year`, `gdp` (constant-price real GDP per capita),
  `Reunification` (treatment indicator; West Germany from 1990 onward).

This is a *teaching / demonstration* dataset, not a validated benchmark.

## Provenance

- West Germany and the 16 donors ADH retained: Francis's (2025) corrected
  constant-price reconstruction — ADH's 1990 level anchored and rescaled by
  World Bank WDI real growth (`NY.GDP.PCAP.KD`), with West Germany's post-1990
  series from the German state accounts (*Volkswirtschaftliche Gesamtrechnungen
  der Länder*). Source: <https://github.com/joefrancis505/Francis_Comparative_Politics>.
  This corrects ADH's use of current-price PPP GDP per capita, the data-handling
  error Francis documents.
- The 7 countries ADH discarded (Canada, Finland, Sweden, Ireland, Luxembourg,
  Iceland, Turkey): Maddison Project Database 2020 (Bolt & van Zanden) real GDP
  per capita, bridged onto the Francis scale via the median 1990 ratio across
  the 16 shared donors (coefficient of variation ≈ 5.6%, so the rescaling is
  close to a constant). These seven series are therefore approximate.

West Germany is West-Germany-only throughout (not unified Germany), so the
post-1990 treated series is not contaminated by the East.
