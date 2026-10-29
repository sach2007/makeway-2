# MANTHAN 2026 Historical Data Inventory

- **Total Platform Assets Ingested**: 10
- **Total Historical Bars**: 6,861
- **Data Format**: Clean point-in-time daily OHLCV (CSV)
- **Storage Directory**: `data/processed/`

### Symbol Summary Table

ticker                           company_name  bars start_date   end_date  min_price  max_price     mean_vol    health
  AERX     AeroVanguard Defense & Space Corp.   687 2022-04-06 2024-12-30     386.29     614.61 1.203630e+06 EXCELLENT
  BLUX     BlueMatrix Enterprise Systems Inc.   686 2022-04-08 2024-12-31     117.57     238.04 4.478315e+06 EXCELLENT
  DSNY      DreamScape Entertainment Universe   686 2022-04-08 2024-12-31      79.32     132.35 1.218285e+07 EXCELLENT
  HAVN HavenStay Travel & Accommodations Inc.   686 2022-04-11 2025-01-02      82.49     171.85 5.675179e+06 EXCELLENT
  HMDP     Haven Depot Home Improvement Corp.   686 2022-04-11 2025-01-02     266.58     431.37 3.625546e+06 EXCELLENT
  MOTR       MotorWorks Global Holdings Corp.   686 2022-04-11 2025-01-02      26.65      60.20 1.502827e+07 EXCELLENT
  ORCX            Oracrest Cloud Systems Inc.   686 2022-04-11 2025-01-02      61.07     192.43 8.544726e+06 EXCELLENT
  PLNX               Polestar Analytics Corp.   686 2022-04-11 2025-01-02       6.00      82.38 5.432677e+07 EXCELLENT
  ROUT   RouteWave Mobility Technologies Inc.   686 2022-04-11 2025-01-02      20.46      86.34 2.372581e+07 EXCELLENT
  SONA        Sonata Digital Audio Media Inc.   686 2022-04-11 2025-01-02      71.05     502.38 1.949039e+06 EXCELLENT

### Data Hygiene & Anti-Lookahead Guarantee
1. **Strict Chronological Sequencing**: All records are strictly ordered by date.
2. **Zero Forward Leakage**: All ingested bars terminate strictly prior to the current simulation clock.
3. **Missing Value Audit**: Zero NaN/null entries across all ingested symbol series.
