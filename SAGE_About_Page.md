
# Welcome to SAGE: Samsung’s Advanced Geoexperiment Tool

SAGE is a system designed to help marketing teams run more reliable geo experiments. It measures true incremental lift, iROAS, and causal impact from campaigns and media investments with greater confidence than traditional approaches.

At its core, SAGE answers a critical question: Which markets should receive the treatment and which should serve as controls so that the experiment produces trustworthy, generalizable results?

---

## Why Geo Experiments Are Challenging

Many geo experiments rely on selecting “similar” markets based on historical performance. While this feels intuitive, it often falls short in practice. Markets that look comparable on baseline metrics can respond quite differently once a campaign begins due to differences in seasonality, competitive environment, customer mix, or underlying sensitivity to marketing. These hidden differences frequently lead to unstable lift estimates and iROAS figures that are difficult to trust or scale.

---

## The Limitations of Traditional Market Selection

When teams choose test and control markets manually, two important issues arise:

1. **The Combinatorial Explosion**: With hundreds of markets, there are billions of ways to assign treatment and control. Human judgment cannot consistently find the "Global Optimum." Traditional tools often take minutes or hours to sample a fraction of these possibilities.
2. **The Power Vacuum**: Even when markets appear similar, there is rarely an objective way to know if the design has enough statistical power to detect the required lift. Many tests end up underpowered, wasting significant time and budget on inconclusive results.

---

## The SAGE Advantage: Principled Optimization

SAGE transforms experimental design from an exercise in intuition into a high-speed optimization task.

### 1. High-Performance Combinatorial Search
SAGE uses advanced **Branch & Bound** algorithms to navigate the search space. Where traditional methods rely on random sampling (Stochastic search), SAGE mathematically prunes the universe of possibilities in seconds to find the provably best market combinations.

### 2. The "Wallet Perspective" (Efficiency Scoring)
SAGE doesn't just look for statistical fit; it looks for **Value**. Every design is evaluated on an **Efficiency Score** that weighs three critical factors:
* **Detection Power (MDE):** Can we actually see the lift if it happens?
* **Stability Ratio:** Does the model remain robust when tested against data it hasn't seen?
* **Budget Utilization:** Are we getting the most "Learning per Dollar," or are we overpaying for marginal gains in precision?

### 3. Synthetic Counterfactuals (The Digital Twin)
Rather than simple pairwise matching, SAGE builds a custom **Digital Twin** of your treatment group using weighted donor markets. This creates a robust national baseline that accounts for non-stationary patterns and autocorrelation, ensuring your iROAS estimates are anchored in reality.

---

## From Design to Reliable Insights

SAGE delivers optimized market assignments, synthetic control weights, and clear confidence intervals. These intervals remain valid even in the complex, "noisy" marketing environments typical of the Samsung ecosystem. The result is not just a point estimate, but a credible range that marketing and finance teams can use to make scaling decisions with confidence.

---

## Why This Matters for the Business

Too often, valuable media budgets are spent on geo tests that were never designed to detect the effects they were meant to measure. SAGE serves as a **Causal Guardrail**, ensuring that every dollar spent on experimentation is a dollar spent on acquiring high-quality business intelligence.

---

## A Note on Usage

SAGE is a **strategic planning and inference engine**. It should be used early in the process to confirm that a design is both financially feasible and statistically capable of delivering the reliable answers the business needs. By making design quality an integral part of the planning process, SAGE ensures your major geo tests are set up for success before the first campaign dollar is committed.
