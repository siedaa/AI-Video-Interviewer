# FirstRound Evals — results



Synthetic persona transcripts scored by src/agents/scorer.py against output/prep/question_plan.json. Run: `python evals/run_evals.py`.

## Ranking table

| Rank | Persona | overall_score | recommendation |
|---|---|---|---|
| 1 | strong | 4.6 | borderline |
| 2 | bluffer | 3.6 | borderline |
| 3 | nervous | 3.4 | borderline |
| 4 | average | 2.6 | no_hire |
| 5 | weak | 1.8 | no_hire |

## Ranking check

Actual order: strong, bluffer, nervous, average, weak
Expected order: strong > nervous > average > bluffer > weak

- Strong lands #1: **PASS**
- Bluffer lands below Average (hard case): **FAIL**
- Nervous lands near Strong (hard case, within one position): **FAIL**
- Weak lands last: **PASS**

All hard cases PASS: False

## Honest analysis



The expected ranking was Strong > Nervous ≈ Average > Bluffer > Weak, with the two hard cases being Bluffer below Average and Nervous near Strong. The scorer put Strong (#1, 4.6), Nervous (#3, 3.4), Average (#4, 2.6), and Weak (last, 1.8) in sensible slots, but **Bluffer misranked at #2 (3.6) — above both Average and Nervous** in both eval runs. This is the exact 'confident and wrong' failure mode the evals were built to expose, and we are reporting it rather than tuning it away.

Why Bluffer passed the scorer: the LLM rewarded fluent, well-structured but substantively empty answers. Despite collapsing on every specific follow-up ('those numbers were tuned by the infra team', 'under NDA', 'I've debugged dozens of these' with no actual method), Bluffer still earned mid-to-high competency scores (Debugging=4, Communication=4). The evidence-check guardrail does not catch this: its quotes are real transcript substrings, so it validates provenance, not correctness — a confident wrong answer is still quotable. Nervous also sits two positions below Strong (rank #3 vs #1), so the 'near Strong' criterion flags; its 3.4 is defensibly 'Nervous ≈ Average, above Average', and the gap is partly an artifact of Bluffer's inflation crowding the top two slots.

Run-to-run variance is real and should be read alongside these numbers: with identical transcripts, run 1 produced strong 4.4/hire, bluffer 4.0/hire, average 3.0/borderline, weak 2.0/no_hire, while run 2 produced 4.6/borderline, 3.6/borderline, 2.6/no_hire, 1.8/no_hire. Competency scores and recommendation come from the LLM (overall_score is a deterministic recomputation of them), so single-run numbers are noisy — but the structural ordering (strong > bluffer > nervous > average > weak) was identical in both runs, so the Bluffer misranking is robust, not a fluke. Oddly, strong rose to 4.6 yet its recommendation dropped to borderline, a symptom of recommendation being LLM-decided rather than derived from overall_score.

Suggested (Phase 8 follow-up, deliberately NOT applied here to avoid hand-tuning): reward answer depth/elaboration and specificity-to-follow-ups in the scoring rubric, derive recommendation from overall_score thresholds instead of leaving it to the LLM, and consider a heuristic guardrail that flags competencies whose evidence_quote is very short or non-technical relative to the transcript's other answers. Weak and Nervous behaved as intended.

## Per-persona detail

- **strong** — overall 4.6, borderline; competencies: Communication=5, Debugging=5, Problem decomposition=5, Shipping ability=3, Technical depth=5; flags: 0
- **bluffer** — overall 3.6, borderline; competencies: Communication=4, Debugging=4, Problem decomposition=4, Shipping ability=3, Technical depth=3; flags: 0
- **nervous** — overall 3.4, borderline; competencies: Communication=3, Debugging=4, Problem decomposition=4, Shipping ability=2, Technical depth=4; flags: 0
- **average** — overall 2.6, no_hire; competencies: Communication=3, Debugging=3, Problem decomposition=3, Shipping ability=2, Technical depth=2; flags: 0
- **weak** — overall 1.8, no_hire; competencies: Communication=2, Debugging=2, Problem decomposition=2, Shipping ability=1, Technical depth=2; flags: 0
