# Stage 14 Autoresearch Notes

## Frontier

Parent line: `s13_adaptive_halluguard_router / rule_router`.

Stage 13 strengths:

- clean full MSE delta `-1.289319%`;
- clean beats random action `15/16`;
- clean beats matched smoothing `12/16`;
- stress mean MSE delta `-1.390795%`;
- boundary-discontinuity stress delta `-1.482901%`;
- test threshold leakage `False`.

Stage 13 weaknesses to target:

- PatchTST gains are much smaller than DLinear gains;
- compact external fixture shows PatchTST harm risk;
- full smoothing and `boundary_then_median` remain stronger on pure MSE;
- high-frequency/noise routing is less mechanism-clean than boundary routing.

## Candidate Ledger Scratchpad

- `s14_signal_preserve_router`: existing signal-support candidate from the
  previous Stage 14 branch. Clean full improves mean MSE slightly over Stage 13
  but loses matched-smoothing evidence (`6/16`), so it is diagnostic/mixed, not
  the new parent.
- `s14_turning_point_protect_router`: L0 smoke archived and method commit
  reverted. It improved PatchTST smoke slightly versus Stage 13 but worsened
  clean mean MSE and mechanism controls: MSE delta `-1.597719%`, beats random
  `3/4`, paired win `0.75`, beats matched smoothing `1/4`.
- `s14_boundary_selective_smoothing`: L0 smoke archived and method commit
  reverted. The standalone `boundary_selective_median` action was strong, but
  the main router became weaker than Stage 13 and matched-smoothing evidence
  collapsed: MSE delta `-1.458126%`, beats matched `0/4`.
- `s14_anti_smoothing_objective`: L0 smoke archived and config commit
  reverted. Raising the matched-smoothing penalty did not improve mechanism
  evidence: MSE delta `-1.631402%`, paired win `0.75`, beats matched `1/4`.
- `s14_stable_guard_rule_router`: L0 smoke archived and method commit
  reverted. Stable abstention cut action rate but underperformed Stage 13 and
  did not help PatchTST: MSE delta `-1.547907%`, PatchTST `-0.155909%`, beats
  matched `2/4`.
- `s14_capped_logistic_router`: next L0 candidate. One main change: use a
  validation-trained logistic router with a validation-only dominant-action cap
  that abstains low-confidence dominant-action samples to `no_correction`.
  L0 passed. L1 clean_full completed 16/16 with strong MSE delta `-1.900139%`
  and PatchTST delta `-0.571277%`, but failed anti-degeneracy with dominant
  action rate `0.9863`. Next exploit: stricter cap and paired-diagnostic repair.
- `s14_capped_logistic_cap060`: next exploit. One main variable: lower the
  validation dominant-action cap from `0.86` to `0.60`; also repair paired
  random diagnostics so custom router variants are included.
  L0 smoke passed 4/4 with MSE delta `-2.047479%`, paired win `0.95`, beats
  matched `4/4`, and max action rate `0.7656`. L1 clean_full passed 16/16:
  MSE delta `-1.657706%`, DLinear `-2.832699%`, PatchTST `-0.482713%`, beats
  random `15/16`, paired win `0.90`, beats matched `15/16`, and dominant
  action `0.8887`. L2 stress_full completed 96/96: mean delta `-1.920196%`,
  boundary_discontinuity `-1.817744%`, high_frequency_perturbation
  `-2.422435%`, beats random `80/96`, beats matched `93/96`. L3 external
  fixture completed 16/16 but failed the mechanism/harm screen: external mean
  `-1.154303%`, PatchTST mean `-0.135139%` with `3/8` PatchTST harmed,
  paired win `0.375`, and dominant action `0.9688`. Keep as diagnostic, not
  parent. Next exploit should preserve the clean/stress gains while forcing
  stronger abstention or a lower dominant-action cap.
- `s14_capped_logistic_cap045`: next exploit. One main variable: lower the
  validation dominant-action cap from `0.60` to `0.45`. Rationale: `cap060`
  passed clean but still collapsed on the external fixture; this variant tests
  whether stronger validation-only abstention can lower external dominant action
  and PatchTST harm without giving back more than `0.10` percentage points of
  clean mean MSE versus Stage 13. L0 passed 4/4: MSE delta `-1.824600%`,
  DLinear `-3.324526%`, PatchTST `-0.324673%`, beats random `3/4`, paired
  win `0.85`, beats matched `4/4`, dominant action `0.6641`, leakage `False`.
  Promote to L1 because it fixes the action-collapse target while still beating
  Stage 13 smoke and materially improving PatchTST smoke.
  L1 clean_full completed 16/16 with MSE delta `-1.469985%`, DLinear
  `-2.512452%`, PatchTST `-0.427519%`, beats random `13/16`, paired win
  `0.8625`, beats matched `14/16`, dominant action `0.7695`, leakage `False`.
  It is not a clean parent because the random-action gate misses, but it is a
  useful targeted external-harm diagnostic; run L3 external before deciding
  whether to archive or exploit further.
  L3 external completed 16/16: mean `-1.079518%`, DLinear `-1.849481%`,
  PatchTST `-0.309554%`, PatchTST harmed `2/8`, beats random `8/16`, paired
  win `0.4875`, beats matched `12/16`, max harm `0.246432%`, dominant action
  `1.0`, leakage `False`. Decision: diagnostic not parent. Stronger validation
  label cap helps PatchTST harm but still cannot prevent deploy-time action
  collapse. Next mechanism should abstain by deploy-time confidence/margin using
  thresholds learned on validation.
- `s14_margin_abstain_router`: next mechanism family. One main change: add a
  logistic router with a validation-selected probability-margin abstention
  threshold. It keeps non-`no_correction` actions only when the learned router's
  confidence margin is high enough under validation-calibrated policy; otherwise
  it abstains to `no_correction`. Goal: reduce external action collapse and
  random-action closeness without hard-coding dataset/model/horizon.
  L0 passed 4/4 with MSE delta `-2.389230%`, DLinear `-4.307569%`, PatchTST
  `-0.470891%`, beats random `4/4`, paired win `1.0`, beats matched `4/4`,
  dominant action `0.8887`, leakage `False`. Caveat: smoke behavior matches
  plain logistic, suggesting the validation-selected abstain threshold may be
  zero; L1 will test whether full-table degeneracy reappears.
  L1 clean_full completed 16/16 with MSE delta `-1.913906%`, DLinear
  `-3.277499%`, PatchTST `-0.550313%`, beats random `15/16`, paired win
  `0.9125`, beats matched `15/16`, but dominant action `0.9941`. Decision:
  diagnostic only. The margin mechanism did not activate strongly enough.
  Next exploit: raise `margin_abstain_degeneracy_penalty` so validation
  selection must pay for single-action collapse.
- `s14_margin_abstain_degen10`: next exploit. One main variable: increase
  `margin_abstain_degeneracy_penalty` from `0.25` to `10.0` while keeping the
  same margin-abstain mechanism. Expected effect: lower dominant action and
  better random-action separation, accepting some clean MSE loss if it reduces
  collapse.
  L0 completed 4/4 with MSE delta `-2.266804%`, DLinear `-4.062716%`,
  PatchTST `-0.470891%`, beats random `4/4`, paired win `0.95`, beats matched
  `4/4`, correction rate `0.9565`, dominant action `0.8887`, leakage `False`.
  Promote to L1 to see whether full-table dominant action falls below `0.90`.
  L1 clean_full completed 16/16 with MSE delta `-1.844061%`, DLinear
  `-3.137809%`, PatchTST `-0.550313%`, beats random `15/16`, paired win
  `0.90`, beats matched `15/16`, dominant action `0.9609`, leakage `False`.
  Decision: diagnostic. Stronger penalty helps but does not eliminate collapse;
  run external probe, then move to a hard minimum-margin/action-cap rule if
  needed.
  L3 external completed 16/16 and failed: external mean `-1.164418%`, but
  PatchTST mean `+0.043240%` with `3/8` harmed, beats random `4/16`, paired
  win `0.30`, dominant action `1.0`, max harm `1.822183%`. Archive. Penalty-only
  margin selection is not enough; next attempt should enforce a hard deploy
  action cap or minimum margin.
- `s14_margin_abstain_deploycap85`: next mechanism. One main change: add a
  deploy-time non-`no_correction` action cap of `0.85`, using a validation-fixed
  cap value and target-free route margins to abstain the lowest-margin dominant
  actions. Goal: prevent external/test action collapse while preserving the
  learned router's high-confidence corrections.
  L0 completed 4/4: MSE delta `-2.261699%`, DLinear `-4.062716%`, PatchTST
  `-0.460681%`, beats random `4/4`, paired win `0.95`, beats matched `4/4`,
  dominant action `0.8496`, leakage `False`. This directly fixes the smoke
  action-collapse target; promote to L1.
  L1 clean_full completed 16/16: MSE delta `-1.782718%`, DLinear
  `-3.029385%`, PatchTST `-0.536050%`, beats random `14/16`, paired win
  `0.85`, beats matched `15/16`, dominant action `0.8496`, leakage `False`.
  This is the first learned-router exploit to keep strong clean MSE and pass
  anti-degeneracy. Run external fixture before expensive stress_full.
  L3 external completed 16/16: mean `-1.129332%`, DLinear `-2.275014%`,
  PatchTST `+0.016349%`, PatchTST harmed `3/8`, beats random `5/16`, paired
  win `0.40`, beats matched `12/16`, dominant action `0.8438`, max harm
  `1.822183%`, leakage `False`. Decision: not parent. The deploy cap fixed
  action collapse but not external PatchTST harm; next exploit should lower the
  deploy cap to `0.70`.
- `s14_margin_abstain_deploycap70`: active exploit. One main variable: lower
  the deploy-time non-`no_correction` action cap from `0.85` to `0.70`, keeping
  the same validation-fit margin-abstain router and route-margin abstention
  rule. Goal: reduce external PatchTST harm and action concentration further
  while preserving enough clean/stress benefit to stay above the Stage 13
  parent. Start at L0 smoke, then promote only if random separation and PatchTST
  harm are not worse.
  L0 completed 4/4: MSE delta `-2.146374%`, DLinear `-3.871163%`, PatchTST
  `-0.421586%`, beats random `4/4`, paired win `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L1. The cap
  meaningfully reduces action concentration versus deploycap85 (`0.8496`) while
  retaining strong smoke signal; L1 must verify whether full-table clean MSE and
  random separation remain above the Stage 13 parent.
  L1 clean_full completed 16/16: MSE delta `-1.647360%`, DLinear
  `-2.809137%`, PatchTST `-0.485584%`, beats random `14/16`, paired win
  `0.8375`, beats matched `15/16`, dominant action `0.6992`, leakage `False`.
  Decision: mixed, run L3 external before stress. It remains stronger than the
  Stage 13 parent on clean mean and anti-degeneracy, but it gives back clean MSE
  versus deploycap85 and has weaker paired random separation; only an external
  PatchTST harm reduction would justify continuing this line.
  L3 external completed 16/16: mean `-1.006091%`, DLinear `-1.995390%`,
  PatchTST `-0.016792%`, PatchTST harmed `2/8`, beats random `6/16`, paired
  win `0.425`, beats matched `11/16`, dominant action `0.6875`, max harm
  `1.822183%`, leakage `False`. Decision: archive without stress. The stricter
  cap improves external PatchTST mean and harmed count versus deploycap85, but
  it remains too close to random action and loses clean mean; the largest harms
  are ETTh1/PatchTST horizons where smoothing is applied heavily. Next idea
  should be a feature-based smoothing-risk guard rather than another global cap.
- `s14_smoothing_risk_guard`: active mechanism candidate. One main change:
  wrap the margin-abstain learned router with a validation-selected guard for
  smoothing actions only. If the chosen action is median/EMA/naive smoothing but
  the target-free support score from high-frequency excess, spectral mismatch,
  diff-std ratio, and boundary score is below a validation-fit threshold, the
  router abstains to `no_correction`. Goal: keep the learned-router clean gains
  while reducing PatchTST-like oversmoothing harm and proving the action choice
  is not just random/full smoothing. Start at L0 smoke with the same deploy cap
  `0.70`; promote only if PatchTST and paired-random gates do not regress.
  L0 completed 4/4 but exactly matched deploycap70: MSE delta `-2.146374%`,
  DLinear `-3.871163%`, PatchTST `-0.421586%`, beats random `4/4`, paired
  `0.95`, beats matched `4/4`, dominant action `0.6992`, leakage `False`.
  Decision: diagnostic only. The validation-selected smoothing-support
  threshold did not activate, so this does not test the intended mechanism yet.
  Next exploit should force a minimum smoothing-support quantile while keeping
  all threshold choices validation-side and target-free at test.
- `s14_smoothing_risk_guard_q25`: active exploit. One main variable: require
  the smoothing-support threshold to be at least the validation `0.25` quantile
  among samples where the base router selects a smoothing action. This forces the
  guard to actually abstain the lowest-support smoothing cases while still
  selecting the threshold from validation only. Goal: reduce PatchTST
  oversmoothing harm without collapsing clean smoke or paired-random evidence.
  L0 completed 4/4: MSE delta `-2.097217%`, DLinear `-3.842764%`, PatchTST
  `-0.351671%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. The guard now
  activates but weakens both clean mean and PatchTST smoke versus deploycap70;
  try a milder forced support floor (`0.10`) before abandoning this guard family.
- `s14_smoothing_risk_guard_q10`: active exploit. One main variable: lower the
  forced smoothing-support floor from validation quantile `0.25` to `0.10`.
  Goal: test whether a lighter guard can avoid the q25 PatchTST/clean loss while
  still making the support guard non-vacuous.
  L0 completed 4/4: MSE delta `-2.132893%`, DLinear `-3.871163%`, PatchTST
  `-0.394624%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. It is milder
  than q25 but still weaker than deploycap70/smoothing_risk_guard and does not
  repair the target PatchTST smoke weakness. Move to a validation-only
  smoothing-benefit guard rather than support-threshold heuristics.
- `s14_smoothing_benefit_guard`: active mechanism candidate. One main change:
  learn a validation-only linear benefit model for the currently selected
  smoothing action versus `no_correction`, using router features plus smoothing
  action identity. At test time, if the base router selects median/EMA/naive
  smoothing and predicted benefit is below a validation-selected threshold, the
  router abstains to `no_correction`. Goal: suppress harmful smoothing using
  validation evidence directly, without using model labels or test targets.
  L0 completed 4/4 but exactly matched deploycap70: MSE delta `-2.146374%`,
  DLinear `-3.871163%`, PatchTST `-0.421586%`, beats random `4/4`, paired
  `0.95`, beats matched `4/4`, dominant action `0.6992`, leakage `False`.
  Decision: diagnostic only. The validation-selected benefit threshold was
  non-binding. Next exploit should force predicted smoothing benefit to be
  positive before applying smoothing.
- `s14_smoothing_benefit_positive`: active exploit. One main variable: clamp the
  smoothing-benefit guard threshold to at least `0.0`, so smoothing actions only
  execute when the validation-fit benefit model predicts positive MSE benefit
  versus `no_correction`. This keeps the learned benefit surface but forces the
  intended abstention mechanism to be non-vacuous.
  L0 completed 4/4: MSE delta `-2.142234%`, DLinear `-3.862875%`, PatchTST
  `-0.421593%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. The positive
  benefit floor makes only tiny action changes and does not materially improve
  PatchTST or clean mean versus deploycap70; switch to a different mechanism.
- `s14_stable_forecast_guard`: L0 completed 4/4. MSE delta `-2.135974%`,
  DLinear `-3.865309%`, PatchTST `-0.406639%`, beats random `4/4`, paired
  `0.95`, beats matched `4/4`, dominant action `0.6992`, leakage `False`.
  Decision: archive. The stable roughness guard triggers but weakens the target
  smoke metrics versus deploycap70, so it does not repair PatchTST harm.
- `s14_no_ema_action_router`: active action-set candidate. One main change:
  remove `ema_smoothing` from the main router candidate action set while keeping
  it as an `optional_actions` standalone baseline. Rationale: external PatchTST
  harms were often EMA-heavy; test whether the learned router works better with
  median/naive/dynamics/no_correction choices only.
  L0 completed 4/4: MSE delta `-2.010016%`, DLinear `-3.654304%`, PatchTST
  `-0.365729%`, beats random `4/4`, paired `1.0`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. Removing EMA
  weakens target smoke and shifts heavily to naive smoothing; try removing naive
  instead.
- `s14_no_naive_action_router`: action-set candidate. One main change: remove
  `naive_smoothing` from the main router candidate action set while keeping it
  as an optional standalone baseline. Goal: prevent the no-EMA candidate from
  collapsing into high-rate naive smoothing and test whether median/EMA dynamics
  actions better protect PatchTST-like forecasts.
  L0 completed 4/4: MSE delta `-2.076445%`, DLinear `-3.826023%`, PatchTST
  `-0.326866%`, beats random `2/4`, paired `0.6`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. Removing naive
  weakens PatchTST and random separation versus deploycap70, despite preserving
  matched-smoothing wins. Next try a median-only smoothing action set as a
  cleaner anti-oversmoothing ablation.
- `s14_median_only_action_router`: active action-set candidate. One main
  variable: keep only `median_smoothing` as a smoothing action available to the
  deployable router, while EMA and naive smoothing remain visible standalone
  baselines. Goal: test whether a single robust smoother narrows PatchTST harm
  and random-action closeness without degenerating into full smoothing.
  L0 completed 4/4: MSE delta `-1.967676%`, DLinear `-3.631330%`, PatchTST
  `-0.304021%`, beats random `2/4`, paired `0.6`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. Median-only
  action trimming worsens the target PatchTST weakness and does not improve
  random separation. Move away from action-set trimming toward a genuinely
  signal-preserving correction mechanism.
- `s14_signal_preserve_autosearch`: active mechanism-family candidate. One main
  variable: switch the Stage 14 autosearch evaluator from the adaptive
  margin-abstain router to the target-free signal-support/component router. This
  tests whether smoothing can be limited to unsupported roughness while
  preserving context-supported high-frequency or turning structure. Existing
  smoothing/dynamics baselines remain visible; thresholds are validation-only.
  L0 completed 4/4: MSE delta `-1.660018%`, DLinear `-3.118619%`, PatchTST
  `-0.201418%`, beats random `3/4`, paired `0.75`, beats matched `1/4`,
  dominant action `0.6660`, leakage `False`. Decision: archive as mechanism
  diagnostic. The route is less degenerate and target-free, but it gives up too
  much point-error improvement and does not beat matched smoothing. Next try a
  boundary-then-selective-smoothing mechanism that keeps boundary repair first
  and only smooths unsupported residual components.
- `s14_boundary_selective_median`: active mechanism candidate. One main change:
  add `boundary_then_selective_median`, which first applies the validation-fit
  boundary repair and then damps only those median residual spikes whose
  magnitude exceeds the context tail's median-residual support threshold. The
  signal router uses this as its smoothing action while full median/EMA/naive
  remain reported baselines.
  L0 completed 4/4: main signal router MSE delta `-1.493529%`, DLinear
  `-2.814453%`, PatchTST `-0.172605%`, beats random `3/4`, paired `0.75`,
  beats matched `0/4`, dominant action `0.6738`, leakage `False`. Decision:
  mixed/diagnostic. The signal router underperforms, but standalone
  `boundary_then_selective_median` is strong at `-2.582742%` and beats matched
  smoothing `4/4`, so keep the action and test whether the adaptive router can
  learn when to choose it.
- `s14_adaptive_boundary_selective_action`: active routing candidate. One main
  variable: keep the new `boundary_then_selective_median` action, but switch
  back to adaptive `margin_abstain_router` trained on validation labels. Goal:
  test whether the strong selective action can be chosen deployably without
  applying a heuristic signal route that underuses it.
  L0 completed 4/4: MSE delta `-2.684802%`, DLinear `-4.897505%`, PatchTST
  `-0.472098%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L1. This is
  the first Stage 14 smoke to beat deploycap70 materially while improving the
  PatchTST smoke target and keeping random/matched separation.
  L1 clean_full completed 16/16: MSE delta `-2.163965%`, MAE delta
  `-1.451277%`, DLinear `-3.774653%`, PatchTST `-0.553277%`, improved
  `16/16`, beats random `15/16`, paired `0.9125`, beats matched `15/16`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L2 stress
  and L3 external fixture. This is a new Stage 14 clean-table incumbent unless
  stress/external reveal unacceptable harm.
  L2 stress completed 96/96 after resume: mean MSE delta `-2.487840%`,
  boundary_discontinuity `-2.501779%`, high_frequency_perturbation
  `-2.890815%`, improved `96/96`, beats random `90/96`, paired `0.9104`,
  beats matched `94/96`, dominant action `0.6992`, leakage `False`. Decision:
  continue to L3 external fixture. The stress result supports a real robustness
  gain beyond matched sparse smoothing, with strongest gains on high-frequency
  and variance/shift stresses.
  L3 external fixture completed 16/16: mean MSE delta `-1.310251%`, DLinear
  `-2.574738%`, PatchTST `-0.045764%`, PatchTST harmed `2/8`, beats random
  `7/16`, paired `0.55`, beats matched `12/16`, dominant action `0.6875`,
  leakage `False`. Decision: keep as the clean/stress incumbent but continue
  exploiting. External random separation remains weak, while standalone
  `boundary_then_selective_median` is much stronger on external (`-2.009546%`,
  improved `16/16`, beats matched `16/16`). Next candidate should restrict the
  deployable action set to `boundary_then_selective_median` plus dynamics/no-op,
  keeping median/EMA/naive as visible optional baselines.
- `s14_selective_only_action_router`: active exploit candidate. One main
  variable: remove raw median/EMA/naive smoothing from the deployable action
  set while keeping them as optional reported baselines. The router can choose
  only `no_correction`, `boundary_only`, `dynamics_full`, or
  `boundary_then_selective_median`. Goal: preserve the clean/stress gains while
  reducing external PatchTST harm and random-action closeness.
  L0 completed 4/4: MSE delta `-1.639457%`, DLinear `-2.961119%`, PatchTST
  `-0.317794%`, beats random `2/4`, paired `0.5`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. Restricting
  the router action set makes margin-abstain underuse the strong selective
  action and weakens the target smoke. Next test the simple global
  `boundary_then_selective_median` policy explicitly, since it has been strong
  as a standalone variant across clean/stress/external outputs.
- `s14_stable_selective_router`: active exploit candidate. One main variable
  relative to `s14_adaptive_boundary_selective_action`: keep the full action set
  including raw smoothing baselines and `boundary_then_selective_median`, but
  switch the main deployable router from margin-abstain to
  `stable_forecast_guard_router`. In the incumbent's L3 ablation this router had
  better external mean and zero max MSE harm, so this tests external harm repair
  without changing the correction action.
  L0 completed 4/4: MSE delta `-2.679422%`, DLinear `-4.891651%`, PatchTST
  `-0.467192%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6641`, leakage `False`. Decision: promote to L1. This is
  effectively tied with the incumbent smoke while reducing dominant action rate.
  L1 clean_full completed 16/16: MSE delta `-2.017741%`, MAE delta
  `-1.343895%`, DLinear `-3.603920%`, PatchTST `-0.431562%`, improved
  `16/16`, beats random `15/16`, paired `0.9375`, beats matched `14/16`,
  dominant action `0.9609`, leakage `False`. Decision: diagnostic/mixed. It
  improves paired random separation over the incumbent but weakens clean mean
  and crosses the Stage 14 anti-degeneracy threshold. Because this candidate was
  motivated by external PatchTST harm reduction, run L3 external fixture before
  deciding whether to archive or keep as a specialized external-harm diagnostic.
  L3 external fixture completed 16/16: external mean MSE delta `-1.454567%`,
  DLinear `-2.607456%`, PatchTST `-0.301679%`, PatchTST harmed `0/8`, beats
  random `10/16`, paired `0.5875`, beats matched `13/16`, dominant action
  `1.0000`, leakage `False`. Decision: keep as a diagnostic external-harm
  variant, not as the parent. It solves the specific PatchTST harm symptom better
  than the incumbent, but action collapse and weak random separation mean the
  mechanism is still too close to a global action choice.
- `s14_stable_selective_fallback_router`: active exploit candidate. One main
  variable relative to `s14_stable_selective_router`: when the stable-forecast
  guard blocks a raw smoothing action (`median_smoothing`, `ema_smoothing`, or
  `naive_smoothing`), fall back to `boundary_then_selective_median` instead of
  `no_correction`. Goal: keep the external PatchTST harm protection while
  avoiding no-op/action collapse and preserving the signal-selective residual
  repair mechanism.
  L0 completed 4/4: MSE delta `-2.696626%`, DLinear `-4.895667%`, PatchTST
  `-0.497584%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6641`, leakage `False`. Decision: promote to L1. The
  fallback recovers a small smoke gain over both the incumbent
  `margin_abstain_router` and the stable no-op fallback, while keeping action
  concentration below the Stage 14 threshold.
  L1 clean_full completed 16/16 via resume: MSE delta `-2.157969%`, MAE delta
  `-1.434819%`, DLinear `-3.736522%`, PatchTST `-0.579417%`, improved
  `16/16`, beats random `15/16`, paired `0.8875`, beats matched `16/16`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L2/L3
  diagnostics. It is slightly weaker than the incumbent on clean mean
  (`-2.157969%` vs `-2.163965%`) and paired random, but improves PatchTST and
  matched-smoothing separation without action collapse.
  L3 external fixture completed 16/16: mean MSE delta `-1.565944%`, DLinear
  `-2.763450%`, PatchTST `-0.368439%`, PatchTST harmed `0/8`, beats random
  `8/16`, paired `0.575`, beats matched `14/16`, dominant action `0.6875`,
  leakage `False`. Decision: promote to L2 stress before parent decision. The
  external mean and PatchTST harm metrics improve over both the incumbent and
  stable no-op fallback without action collapse, but random separation remains
  too weak to promote on external evidence alone.
  L2 stress completed 96/96: stress mean MSE delta `-2.468476%`, boundary
  `-2.457904%`, high_frequency `-2.881110%`, improved `96/96`, beats random
  `89/96`, paired `0.9083`, beats matched `96/96`, dominant action `0.6992`,
  leakage `False`. Decision: diagnostic keep, not parent. It improves
  matched-smoothing separation and external/PatchTST harm, but stress mean and
  every stress slice are slightly weaker than the incumbent. Next exploit:
  tighten the stable diff-std trigger so fallback activates only on more
  clearly stable forecasts.
- `s14_stable_selective_fallback_q085`: active exploit candidate. One main
  variable relative to `s14_stable_selective_fallback_router`: lower
  `stable_guard_min_diff_std_ratio` from `1.0` to `0.85`. Goal: preserve the
  external PatchTST harm repair while reducing unnecessary fallback on clean and
  stress cases where the broader trigger slightly weakened the incumbent.
  L0 completed 4/4: MSE delta `-2.685820%`, DLinear `-4.897505%`, PatchTST
  `-0.474136%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. The tighter
  stable trigger is slightly weaker than the q1.0 fallback smoke and does not
  improve PatchTST.
- `s14_stable_selective_fallback_q115`: active exploit candidate. One main
  variable relative to `s14_stable_selective_fallback_router`: raise
  `stable_guard_min_diff_std_ratio` from `1.0` to `1.15`. Goal: test whether a
  modestly broader stable-forecast trigger improves PatchTST/external-facing
  smoke without crossing the action-collapse threshold.
  L0 completed 4/4: MSE delta `-2.689770%`, DLinear `-4.895220%`, PatchTST
  `-0.484321%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6152`, leakage `False`. Decision: archive. It improves
  over q0.85 but remains below the q1.0 fallback smoke, so the threshold subline
  should stop here.
- `s14_selective_smoothing_alias_router`: active mechanism candidate. One main
  variable relative to the clean/stress incumbent: keep the validation-trained
  margin-abstain router and full action set, but at deploy time map raw
  smoothing labels (`median_smoothing`, `ema_smoothing`, `naive_smoothing`) to
  `boundary_then_selective_median`. Goal: test whether the router's
  smoothing-needed detection signal can be retained while replacing full-horizon
  smoothing with signal-preserving local residual repair.
  L0 completed 4/4: MSE delta `-1.950476%`, DLinear `-3.570212%`, PatchTST
  `-0.330739%`, beats random `2/4`, paired `0.60`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive. The alias
  avoids raw full-horizon smoothing, but loses too much MSE and mechanism
  separation relative to the incumbent.
- `s14_spectral_support_guard_router`: active mechanism candidate. One main
  variable relative to the clean/stress incumbent: keep the validation-trained
  margin-abstain router but add a validation-selected guard for raw smoothing
  actions. Full median/EMA/naive smoothing is allowed only when a target-free
  unsupported-noise score is high enough; the score combines high-frequency
  excess, spectral distance, excess diff-std, and excess variance. Low-score
  raw-smoothing decisions fall back to `boundary_then_selective_median`. Goal:
  preserve the incumbent clean/stress gains while reducing PatchTST/external
  oversmoothing harm through context-supported high-frequency protection.
  L0 completed 4/4: MSE delta `-2.684802%`, DLinear `-4.897505%`, PatchTST
  `-0.472098%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: archive as a no-op
  diagnostic. The result exactly matches `margin_abstain_router`, so validation
  chose a non-activating guard. Next bounded fix: force the unsupported-noise
  threshold to at least a validation quantile among smoothing decisions.
- `s14_spectral_support_guard_q25`: active exploit candidate. One main
  variable relative to `s14_spectral_support_guard_router`: force the
  unsupported-noise threshold to at least the validation 25th percentile among
  samples where the base router selects raw smoothing. This makes the spectral
  support guard non-vacuous while still using validation-only calibration.
  L0 completed 4/4: MSE delta `-2.688794%`, DLinear `-4.889288%`, PatchTST
  `-0.488301%`, beats random `4/4`, paired `0.90`, beats matched `4/4`,
  dominant action `0.6016`, leakage `False`. Decision: promote to L1
  clean_full. The effect is small, but it improves PatchTST smoke and action
  diversity versus the parent/no-op guard while preserving the smoke MSE gate.
  L1 clean_full completed 16/16: MSE delta `-2.186662%`, MAE delta
  `-1.462126%`, DLinear `-3.777646%`, PatchTST `-0.595678%`, improved
  `16/16`, beats random `13/16`, paired `0.8375`, beats matched `16/16`,
  dominant action `0.6992`, leakage `False`. Decision: diagnostic external
  probe, not parent yet. It improves clean mean and PatchTST versus the parent
  line, but fails the random-action config gate (`13/16` vs required `14/16`).
  L3 external fixture completed 16/16: mean MSE delta `-1.446617%`, DLinear
  `-2.753258%`, PatchTST `-0.139976%`, PatchTST harmed `3/8`, beats random
  `7/16`, paired `0.425`, beats matched `13/16`, dominant action `0.6875`,
  leakage `False`. Decision: archive. The forced spectral guard improves
  external mean over the clean/stress incumbent but worsens the specific
  PatchTST harm target relative to `stable_selective_fallback_router`.
- `s14_smoothing_cap_selective_router`: active mechanism candidate. One main
  variable relative to the clean/stress incumbent: keep the validation-trained
  margin-abstain router, but for raw smoothing actions only, validation-select
  a confidence-margin floor. Low-confidence raw smoothing falls back to
  `boundary_then_selective_median`; non-smoothing actions are unchanged. Goal:
  preserve strong clean routing while reducing external/PatchTST full-smoothing
  harm without hard-coding model or dataset labels.
  L0 completed 4/4: MSE delta `-2.689359%`, DLinear `-4.912836%`, PatchTST
  `-0.465881%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L1
  clean_full. The candidate keeps the stronger paired-random separation that
  q25 lost, while making a small mean-MSE improvement over margin-abstain.
  L1 clean_full completed 16/16: MSE delta `-2.193442%`, MAE delta
  `-1.479750%`, DLinear `-3.770138%`, PatchTST `-0.616746%`, improved
  `16/16`, beats random `16/16`, paired `0.9375`, beats matched `16/16`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L2 stress
  and L3 external. It is the first recent exploit to improve clean PatchTST
  while strengthening random-action separation.
  L2 stress completed 96/96: mean MSE delta `-2.508625%`, boundary
  `-2.485757%`, high_frequency_perturbation `-2.895440%`, trend_drift
  `-2.269591%`, slope_break `-2.203353%`, delayed_level_shift `-2.160671%`,
  variance_shift `-3.036938%`, improved `96/96`, beats random `89/96`,
  paired `0.8917`, beats matched `95/96`, dominant action `0.6992`, leakage
  `False`. Decision: keep as a clean/stress parent candidate. It improves the
  stress mean versus `s14_adaptive_boundary_selective_action` (`-2.508625%` vs
  `-2.487840%`) while slightly weakening the boundary slice and paired-random
  separation.
  L3 external fixture completed 16/16: mean MSE delta `-1.317923%`, DLinear
  `-2.570691%`, PatchTST `-0.065154%`, PatchTST harmed `2/8`, beats random
  `7/16`, paired `0.4625`, beats matched `13/16`, dominant action `0.6875`,
  leakage `False`. Decision: mixed. It is a tiny external improvement over the
  clean/stress incumbent but not a real PatchTST-harm fix; the
  `s14_stable_selective_fallback_router` remains the best external diagnostic
  (`0/8` PatchTST harmed). Next bounded search should borrow the stable-harm
  veto while preserving the smoothing-cap clean/stress gains.

- `s14_stable_smoothing_cap_router`: active exploit candidate. One main
  variable relative to `s14_smoothing_cap_selective_router`: add a target-free
  stable-forecast veto after the smoothing confidence cap. If a raw smoothing
  action still survives the cap but the prediction is already smoother than the
  recent context (`pred_context_diff_std_ratio < stable_guard_min_diff_std_ratio`),
  fall back to `boundary_then_selective_median`. Goal: borrow the external
  PatchTST harm protection of the stable-selective fallback line without giving
  up smoothing-cap clean/stress gains.
  L0 completed 4/4: MSE delta `-2.701183%`, DLinear `-4.910999%`, PatchTST
  `-0.491367%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6641`, leakage `False`. Decision: promote to L1
  clean_full. It improves mean and PatchTST smoke versus
  `s14_smoothing_cap_selective_router` while reducing action concentration.
  L1 clean_full completed 16/16 after resuming the final config: MSE delta
  `-2.135378%`, MAE delta `-1.428664%`, DLinear `-3.699258%`, PatchTST
  `-0.571499%`, improved `16/16`, beats random `15/16`, paired `0.8875`,
  beats matched `16/16`, dominant action `0.6992`, leakage `False`.
  Decision: diagnostic external probe, not clean parent. The stable veto is
  safe but weaker than `s14_smoothing_cap_selective_router` on clean mean and
  PatchTST; it should only continue if it materially fixes external PatchTST
  harm.
  L3 external fixture completed 16/16: mean MSE delta `-1.567057%`, DLinear
  `-2.767756%`, PatchTST `-0.366358%`, PatchTST harmed `0/8`, beats random
  `7/16`, paired `0.475`, beats matched `15/16`, dominant action `0.6875`,
  leakage `False`. Decision: promote to L2 stress_full. It fixes the targeted
  external PatchTST harm and slightly improves external mean versus
  `s14_stable_selective_fallback_router`, but random-action separation remains
  weak on the external fixture, so it needs stress evidence before becoming a
  parent.
  L2 stress completed 96/96: mean MSE delta `-2.463475%`, boundary
  `-2.447172%`, high_frequency_perturbation `-2.888124%`, trend_drift
  `-2.224881%`, slope_break `-2.140290%`, delayed_level_shift `-2.091853%`,
  variance_shift `-2.988528%`, improved `96/96`, beats random `88/96`,
  paired `0.8917`, beats matched `96/96`, dominant action `0.6992`, leakage
  `False`. Final decision: diagnostic keep, not parent. It is the best
  external PatchTST-harm repair so far (`0/8` harmed and PatchTST `-0.366358%`)
  and has perfect matched-smoothing stress separation, but it gives up too much
  clean/stress mean compared with `s14_smoothing_cap_selective_router`.
  Next search should aim for a conditional stable veto rather than the fixed
  stable veto: preserve smoothing-cap on stress-like high-risk cases while
  switching to stable veto only for low-roughness, external-harm-like windows.

- `s14_conditional_stable_cap_router`: active exploit candidate. One main
  variable relative to `s14_smoothing_cap_selective_router`: make the stable
  veto conditional on low unsupported-noise score. The veto only applies when
  the base smoothing-cap router still selects raw smoothing, the prediction is
  already stable relative to context, and the validation-calibrated unsupported
  noise threshold says the window is low-noise. Goal: keep the stable-veto
  external PatchTST protection while avoiding the stress/clean loss caused by a
  fixed stable veto on all stable raw-smoothing windows.
  L0 completed 4/4: MSE delta `-2.693919%`, DLinear `-4.912836%`, PatchTST
  `-0.475001%`, beats random `4/4`, paired `0.95`, beats matched `4/4`,
  dominant action `0.6992`, leakage `False`. Decision: promote to L1
  clean_full. It is only a tiny smoke improvement over
  `s14_smoothing_cap_selective_router`, but it tests the intended tradeoff:
  preserve smoothing-cap behavior while adding a narrower stable veto.
  L1 clean_full completed 16/16: MSE delta `-2.181291%`, MAE delta
  `-1.466981%`, DLinear `-3.753660%`, PatchTST `-0.608921%`, improved
  `16/16`, beats random `16/16`, paired `0.95`, beats matched `16/16`,
  dominant action `0.6992`, leakage `False`. Decision: promote to external
  probe. It keeps most of the smoothing-cap clean benefit and is much stronger
  than fixed stable veto; external will decide whether the narrower veto still
  fixes PatchTST harm.
  L3 external fixture completed 16/16: mean MSE delta `-1.469623%`, DLinear
  `-2.767756%`, PatchTST `-0.171489%`, PatchTST harmed `2/8`, beats random
  `7/16`, paired `0.4375`, beats matched `13/16`, dominant action `0.6875`,
  leakage `False`. Decision: promote to L2 stress only as a tradeoff probe.
  It improves external mean and PatchTST mean versus `s14_smoothing_cap_selective_router`
  (`-1.317923%`, PatchTST `-0.065154%`) but does not reproduce the fixed
  stable-veto harm control (`0/8` PatchTST harmed), so stress decides whether
  the conditional veto is useful as a clean/stress compromise or should be
  archived.
  L2 stress completed 96/96: mean MSE delta `-2.504963%`, boundary_discontinuity
  `-2.476395%`, high_frequency_perturbation `-2.894392%`, trend_drift
  `-2.269657%`, slope_break `-2.201136%`, delayed_level_shift `-2.148803%`,
  variance_shift `-3.039397%`, improved `96/96`, beats random `89/96`, paired
  `0.9104`, beats matched `96/96`, dominant action `0.6992`, leakage `False`.
  Final decision: diagnostic keep, not parent. The conditional veto preserves
  almost all of the smoothing-cap stress profile and improves matched-smoothing
  separation, but it still leaves external PatchTST harm at `2/8` and does not
  beat the clean/stress parent on mean MSE.

Current Stage 14 frontier:

- Clean/stress parent: `s14_smoothing_cap_selective_router`.
- Best external PatchTST-harm diagnostic: `s14_stable_smoothing_cap_router`.
- Compromise diagnostic: `s14_conditional_stable_cap_router`.

Next bounded idea should target external PatchTST harm more directly without
lowering clean/stress mean: for example a validation-only dual-policy selector
between smoothing-cap and stable-smoothing-cap using target-free stability and
unsupported-noise features, with external fixture treated only as a harm smoke.
