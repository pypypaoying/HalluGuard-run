# Stage 14 Autoresearch Report

Current incumbent: `s13_adaptive_halluguard_router / rule_router`.

The Stage 14 search loop is now organized under:

- `experiments/halluguard/run_stage14_autosearch.py`
- `experiments/halluguard/configs/halluguard_stage14_autosearch.yaml`
- `experiments/halluguard/results/stage14_autosearch/results.tsv`

The prior `s14_signal_preserve_router` branch result is treated as a mixed
diagnostic candidate: it slightly improved clean mean MSE over Stage 13 but did
not improve the matched-smoothing mechanism gate.

## Tried Candidates

- `s14_turning_point_protect_router`: archived after L0 smoke. PatchTST smoke
  improved slightly, but clean mean MSE delta (`-1.597719%`) was weaker than
  Stage 13 on the same smoke (`-1.744032%`), paired random win rate was `0.75`,
  and matched-smoothing wins were only `1/4`. The method commit was reverted.
- `s14_boundary_selective_smoothing`: archived after L0 smoke. The standalone
  `boundary_selective_median` action was strong (`-2.122182%` smoke MSE delta),
  but the main router using selective smoothing fell to `-1.458126%`, below
  Stage 13 smoke, and beat matched smoothing in `0/4` configs. The method commit
  was reverted.
- `s14_anti_smoothing_objective`: archived after L0 smoke. Increasing the
  validation matched-smoothing penalty did not rescue the mechanism gate:
  main MSE delta was `-1.631402%`, paired random win rate was `0.75`, and
  matched-smoothing wins stayed at `1/4`. The config commit was reverted.
- `s14_stable_guard_rule_router`: archived after L0 smoke. The stable abstain
  guard made the router more conservative but weaker: MSE delta `-1.547907%`
  versus Stage 13 smoke `-1.744032%`, PatchTST `-0.155909%`, and matched wins
  `2/4`. The method commit was reverted.

## Active Candidate

`s14_capped_logistic_router` revisits the learned-router family with an explicit
validation-only dominant-action cap. It keeps the strong L0 logistic signal but
tries to prevent the full-table degeneracy that made learned routers unsuitable
in Stage 13.

L1 clean_full result: mixed. The candidate improved clean mean MSE to
`-1.900139%` and PatchTST mean to `-0.571277%`, but max dominant action rate was
`0.9863`, so it is too close to a single-action learned smoother. It remains a
diagnostic lead, not the parent.

## Active Exploit

`s14_capped_logistic_cap060` lowers the validation dominant-action cap to `0.60`
and repairs paired-random diagnostics for custom router variants. It tests
whether the learned-router MSE signal survives a much stricter anti-degeneracy
constraint.

Results:

- L0 smoke completed `4/4`: MSE delta `-2.047479%`, DLinear `-3.699620%`,
  PatchTST `-0.395338%`, beats random action `4/4`, paired win `0.95`, beats
  matched smoothing `4/4`, max dominant action `0.7656`, leakage `False`.
- L1 clean_full completed `16/16`: MSE delta `-1.657706%`, DLinear
  `-2.832699%`, PatchTST `-0.482713%`, improved `16/16`, beats random action
  `15/16`, paired win `0.90`, beats matched smoothing `15/16`, max dominant
  action `0.8887`, leakage `False`.
- L2 stress_full completed `96/96`: mean MSE delta `-1.920196%`,
  boundary_discontinuity `-1.817744%`, high_frequency_perturbation
  `-2.422435%`, beats random action `80/96`, paired win `0.8271`, beats
  matched smoothing `93/96`, leakage `False`. The high-frequency stress slice
  reaches dominant action `0.9199`, so it is not cleanly non-degenerate.
- L3 external fixture completed `16/16`: mean MSE delta `-1.154303%`, DLinear
  `-2.173467%`, PatchTST `-0.135139%`, PatchTST harmed `3/8`, beats random
  action only `6/16`, paired win `0.375`, max MSE harm `0.804107%`, dominant
  action `0.9688`, leakage `False`.

Decision: diagnostic keep, not parent promotion. The stricter cap fixes the
clean full-table degeneracy and improves PatchTST relative to Stage 13, but the
external fixture shows the router still behaves too much like a high-rate
smoothing selector and is not sufficiently separated from matched random action.
The next one-variable exploit should tighten abstention/action cap further or
make the cap act on the deploy-time action histogram rather than only the
validation training labels.

## Follow-Up Exploit: `s14_capped_logistic_cap045`

`s14_capped_logistic_cap045` tightens the validation dominant-action cap from
`0.60` to `0.45`.

Results:

- L0 smoke completed `4/4`: MSE delta `-1.824600%`, DLinear `-3.324526%`,
  PatchTST `-0.324673%`, beats random action `3/4`, paired win `0.85`, beats
  matched `4/4`, dominant action `0.6641`, leakage `False`.
- L1 clean_full completed `16/16`: MSE delta `-1.469985%`, DLinear
  `-2.512452%`, PatchTST `-0.427519%`, beats random action `13/16`, paired
  win `0.8625`, beats matched `14/16`, dominant action `0.7695`, leakage
  `False`.
- L3 external fixture completed `16/16`: mean MSE delta `-1.079518%`, DLinear
  `-1.849481%`, PatchTST `-0.309554%`, PatchTST harmed `2/8`, beats random
  action `8/16`, paired win `0.4875`, max MSE harm `0.246432%`, dominant action
  `1.0`, leakage `False`.

Decision: archive as diagnostic, not parent. The stricter validation cap
improves PatchTST external harm relative to `cap060`, but deploy-time action
collapse persists and random-action separation remains too weak. The next
candidate should add a validation-fit confidence/margin abstention rule at
application time, rather than only altering validation label caps.

## Margin-Abstain / Deploy-Cap Line

`s14_margin_abstain_router` added a validation-selected confidence-margin
abstention rule. It found a strong clean signal but failed the anti-collapse
screen: L1 clean_full MSE delta `-1.913906%`, PatchTST `-0.550313%`, beats
random `15/16`, beats matched `15/16`, but dominant action `0.9941`.

`s14_margin_abstain_degen10` increased the validation degeneracy penalty to
`10.0`. L1 stayed strong (`-1.844061%`, PatchTST `-0.550313%`) but remained too
concentrated (`0.9609`), and L3 external failed with PatchTST mean `+0.043240%`
and `3/8` harmed.

`s14_margin_abstain_deploycap85` added a target-free deploy-time cap that
abstains lowest-margin dominant actions. L1 clean_full passed anti-degeneracy:
MSE `-1.782718%`, DLinear `-3.029385%`, PatchTST `-0.536050%`, beats random
`14/16`, paired `0.85`, beats matched `15/16`, dominant `0.8496`. External
still failed the PatchTST harm diagnostic: PatchTST mean `+0.016349%`,
`3/8` harmed, paired random win `0.40`.

`s14_margin_abstain_deploycap70` tightens only that deploy-time cap from `0.85`
to `0.70`.

Current result:

- L0 smoke completed `4/4`: MSE delta `-2.146374%`, DLinear `-3.871163%`,
  PatchTST `-0.421586%`, beats random `4/4`, paired `0.95`, beats matched
  `4/4`, dominant action `0.6992`, leakage `False`.
- L1 clean_full completed `16/16`: MSE delta `-1.647360%`, DLinear
  `-2.809137%`, PatchTST `-0.485584%`, beats random `14/16`, paired `0.8375`,
  beats matched `15/16`, beats boundary_only `16/16`, dominant action `0.6992`,
  leakage `False`.
- L3 external fixture completed `16/16`: mean MSE delta `-1.006091%`, DLinear
  `-1.995390%`, PatchTST `-0.016792%`, PatchTST harmed `2/8`, beats random
  action `6/16`, paired win `0.425`, beats matched `11/16`, dominant action
  `0.6875`, max harm `1.822183%`, leakage `False`.

Decision: archive without stress. The cap70 variant stays above the Stage 13
rule-router clean mean and keeps action concentration healthy, and it improves
external PatchTST harm slightly versus deploycap85. But the external random
separation remains too weak and random_action_router slightly beats the main
mean on the fixture. The cap-only exploit line is not enough; the next candidate
should target smoothing-risk features directly, especially ETTh1/PatchTST-like
cases where aggressive smoothing causes the largest harm.

## Smoothing-Risk Guard Line

`s14_smoothing_risk_guard` adds a validation-selected smoothing-support guard
around the margin-abstain router. Smoothing actions are supposed to fire only
when a target-free support score from high-frequency excess, spectral mismatch,
diff-std ratio, and boundary score clears a validation-fit threshold.

L0 smoke completed `4/4`, but the result exactly matched
`s14_margin_abstain_deploycap70`: MSE delta `-2.146374%`, DLinear
`-3.871163%`, PatchTST `-0.421586%`, beats random `4/4`, paired `0.95`, beats
matched `4/4`, dominant action `0.6992`, leakage `False`.

Decision: diagnostic only, no L1 promotion. The validation objective selected a
non-binding support threshold, so the intended smoothing-risk mechanism was not
actually tested. The next exploit should force a minimum validation support
quantile for smoothing actions while preserving validation-only calibration.

`s14_smoothing_risk_guard_q25` forced that minimum support threshold to the
validation 25th percentile among smoothing-selected samples. L0 smoke completed
`4/4`: MSE delta `-2.097217%`, DLinear `-3.842764%`, PatchTST `-0.351671%`,
beats random `4/4`, paired `0.95`, beats matched `4/4`, dominant action
`0.6992`, leakage `False`.

Decision: archive. The guard became non-vacuous but weakened the target
PatchTST smoke and clean mean relative to deploycap70. A milder q10 floor is the
last cheap exploit for this feature family; if it also fails, the search should
move away from support-threshold guards.

`s14_smoothing_risk_guard_q10` lowered the forced support floor to the
validation 10th percentile. L0 smoke completed `4/4`: MSE delta `-2.132893%`,
DLinear `-3.871163%`, PatchTST `-0.394624%`, beats random `4/4`, paired
`0.95`, beats matched `4/4`, dominant action `0.6992`, leakage `False`.

Decision: archive. The milder support threshold recovers part of q25's loss but
still underperforms deploycap70 and does not improve PatchTST smoke. The
support-threshold guard family did not solve the harm problem; the next line
should predict validation smoothing benefit directly.

`s14_smoothing_benefit_guard` replaces the heuristic support threshold with a
validation-only linear model of selected smoothing action benefit versus
`no_correction`. L0 smoke completed `4/4`, but exactly matched deploycap70:
MSE delta `-2.146374%`, DLinear `-3.871163%`, PatchTST `-0.421586%`, beats
random `4/4`, paired `0.95`, beats matched `4/4`, dominant action `0.6992`,
leakage `False`.

Decision: diagnostic only. The validation-selected benefit threshold was
non-binding. The next single-variable test should force positive predicted
smoothing benefit before applying smoothing, to make the learned benefit guard
non-vacuous.

## Boundary Selective Action Line

`s14_adaptive_boundary_selective_action` is the current clean/stress incumbent.
It adds `boundary_then_selective_median` as a deployable action and routes over
the full action set with validation-only margin abstention.

Best completed result so far:

- L1 clean_full completed `16/16`: MSE delta `-2.163965%`, DLinear
  `-3.774653%`, PatchTST `-0.553277%`, beats random `15/16`, paired `0.9125`,
  beats matched `15/16`, dominant action `0.6992`, leakage `False`.
- L2 stress completed `96/96`: stress mean MSE delta `-2.487840%`,
  boundary_discontinuity `-2.501779%`, high_frequency_perturbation
  `-2.890815%`, beats random `90/96`, beats matched `94/96`, leakage `False`.
- L3 external fixture completed `16/16`: mean MSE delta `-1.310251%`, DLinear
  `-2.574738%`, PatchTST `-0.045764%`, PatchTST harmed `2/8`, beats random
  `7/16`, paired `0.55`, beats matched `12/16`.

`s14_stable_selective_router` kept the same action set but changed only the main
router to `stable_forecast_guard_router`. L1 clean_full completed `16/16` with
MSE delta `-2.017741%`, DLinear `-3.603920%`, PatchTST `-0.431562%`, beats
random `15/16`, paired `0.9375`, beats matched `14/16`, but dominant action
rose to `0.9609`. L3 external fixture completed `16/16` with mean MSE delta
`-1.454567%`, DLinear `-2.607456%`, PatchTST `-0.301679%`, PatchTST harmed
`0/8`, beats random `10/16`, paired `0.5875`, beats matched `13/16`, dominant
action `1.0000`, leakage `False`.

Decision: keep `s14_stable_selective_router` as a diagnostic external-harm
variant only. It fixes the PatchTST fixture harm symptom, but it collapses to a
single dominant action and does not provide enough random-action separation to
be the Stage 14 parent. The next candidate should preserve the incumbent's
clean/stress mechanism while borrowing only the stable guard's harm-reduction
idea, ideally through a local abstain/cap rather than a global action choice.

`s14_stable_selective_fallback_router` keeps the stable guard but changes only
the blocked-action fallback: stable/low-roughness forecasts that would have been
sent to raw smoothing now fall back to `boundary_then_selective_median` instead
of `no_correction`.

Current result:

- L0 smoke completed `4/4`: MSE delta `-2.696626%`, DLinear `-4.895667%`,
  PatchTST `-0.497584%`, beats random `4/4`, paired `0.95`, beats matched
  `4/4`, dominant action `0.6641`, leakage `False`.
- L1 clean_full completed `16/16` via resume: MSE delta `-2.157969%`, DLinear
  `-3.736522%`, PatchTST `-0.579417%`, beats random `15/16`, paired `0.8875`,
  beats matched `16/16`, dominant action `0.6992`, leakage `False`.

Decision: promote to stress/external diagnostics. It does not beat the incumbent
on clean mean (`-2.157969%` vs `-2.163965%`) and loses some paired random
separation, but it improves PatchTST and matched-smoothing separation without
the stable no-op variant's action collapse.

External fixture completed `16/16`: mean MSE delta `-1.565944%`, DLinear
`-2.763450%`, PatchTST `-0.368439%`, PatchTST harmed `0/8`, beats random
`8/16`, paired `0.575`, beats matched `14/16`, dominant action `0.6875`,
leakage `False`.

Decision update: continue to L2 stress. The fallback variant improves the
targeted external PatchTST harm and mean MSE without collapsing to one action,
but it still has weak random-action separation on the external fixture, so it
needs stress evidence before becoming the Stage 14 parent.

Stress completed `96/96`: mean MSE delta `-2.468476%`, boundary_discontinuity
`-2.457904%`, high_frequency_perturbation `-2.881110%`, trend_drift
`-2.249408%`, slope_break `-2.173869%`, delayed_level_shift `-2.122349%`,
variance_shift `-2.926218%`, beats random `89/96`, paired `0.9083`, beats
matched `96/96`, dominant action `0.6992`, leakage `False`.

Final decision for this candidate: diagnostic keep, not parent. It is the best
Stage 14 line for external PatchTST harm so far and improves matched-smoothing
separation on stress, but the current clean/stress parent
`s14_adaptive_boundary_selective_action` remains slightly stronger on clean mean
and every stress slice. The next bounded exploit is a stricter stable
diff-std trigger, so fallback only activates on more clearly stable forecasts.

`s14_stable_selective_fallback_q085` lowered the stable diff-std threshold from
`1.0` to `0.85`. L0 smoke completed `4/4`: MSE delta `-2.685820%`, DLinear
`-4.897505%`, PatchTST `-0.474136%`, beats random `4/4`, paired `0.95`, beats
matched `4/4`, dominant action `0.6992`, leakage `False`.

Decision: archive. Tightening the trigger was slightly weaker than q1.0 on
smoke and did not improve PatchTST, so it should not consume a full table.

`s14_stable_selective_fallback_q115` raised the stable diff-std threshold from
`1.0` to `1.15`. L0 smoke completed `4/4`: MSE delta `-2.689770%`, DLinear
`-4.895220%`, PatchTST `-0.484321%`, beats random `4/4`, paired `0.95`, beats
matched `4/4`, dominant action `0.6152`, leakage `False`.

Decision: archive and end this threshold subline. It is better than q0.85 but
still below the q1.0 fallback smoke, so the next search step should change the
mechanism rather than keep moving this threshold.

`s14_selective_smoothing_alias_router` kept the validation-trained
margin-abstain router but mapped raw smoothing deploy actions to
`boundary_then_selective_median`. L0 smoke completed `4/4`: MSE delta
`-1.950476%`, DLinear `-3.570212%`, PatchTST `-0.330739%`, beats random `2/4`,
paired `0.60`, beats matched `4/4`, dominant action `0.6992`, leakage `False`.

Decision: archive. The idea separates smoothing-needed detection from raw
smoothing deployment, but the resulting route is much weaker than the incumbent
and random separation drops too far.

## Smoothing Cap Selective Router

`s14_smoothing_cap_selective_router` keeps the validation-trained
margin-abstain router but changes only raw smoothing deployment: if the selected
raw smoothing action has low validation-calibrated confidence margin, it falls
back to `boundary_then_selective_median`; dynamics and no-correction actions are
unchanged.

Completed evidence:

- L0 smoke completed `4/4`: MSE delta `-2.689359%`, DLinear `-4.912836%`,
  PatchTST `-0.465881%`, beats random `4/4`, paired `0.95`, beats matched
  `4/4`, dominant action `0.6992`, leakage `False`.
- L1 clean_full completed `16/16`: MSE delta `-2.193442%`, MAE delta
  `-1.479750%`, DLinear `-3.770138%`, PatchTST `-0.616746%`, beats random
  `16/16`, paired `0.9375`, beats matched `16/16`, max MSE harm
  `-0.387360%`, leakage `False`.
- L2 stress completed `96/96`: mean MSE delta `-2.508625%`,
  boundary_discontinuity `-2.485757%`, high_frequency_perturbation
  `-2.895440%`, variance_shift `-3.036938%`, beats random `89/96`, paired
  `0.8917`, beats matched `95/96`, max MSE harm `-0.365759%`, leakage
  `False`.
- L3 external fixture completed `16/16`: mean MSE delta `-1.317923%`, DLinear
  `-2.570691%`, PatchTST `-0.065154%`, PatchTST harmed `2/8`, beats random
  `7/16`, paired `0.4625`, beats matched `13/16`, max MSE harm `1.743747%`,
  leakage `False`.

Decision: keep as the best clean/stress parent candidate so far, but do not
call it an external-harm repair. It improves clean mean, clean PatchTST, and
stress mean versus `s14_adaptive_boundary_selective_action`, while external
PatchTST harm is only slightly better than parent and clearly worse than the
diagnostic `s14_stable_selective_fallback_router`. The next candidate should
make one targeted change: add a stable-forecast harm veto to smoothing-cap
decisions, using validation-only thresholds and no model/dataset shortcut.

## Stable Smoothing Cap Router

`s14_stable_smoothing_cap_router` adds one stable-forecast veto after the
smoothing confidence cap: if raw smoothing survives the cap but the prediction
diff-std is below the context diff-std threshold, the action falls back to
`boundary_then_selective_median`.

Completed evidence:

- L0 smoke completed `4/4`: MSE delta `-2.701183%`, PatchTST `-0.491367%`,
  beats random `4/4`, paired `0.95`, beats matched `4/4`, dominant action
  `0.6641`, leakage `False`.
- L1 clean_full completed `16/16`: MSE delta `-2.135378%`, DLinear
  `-3.699258%`, PatchTST `-0.571499%`, beats random `15/16`, paired `0.8875`,
  beats matched `16/16`, leakage `False`.
- L3 external fixture completed `16/16`: mean MSE delta `-1.567057%`, DLinear
  `-2.767756%`, PatchTST `-0.366358%`, PatchTST harmed `0/8`, beats matched
  `15/16`, leakage `False`.
- L2 stress completed `96/96`: mean MSE delta `-2.463475%`,
  boundary_discontinuity `-2.447172%`, high_frequency_perturbation
  `-2.888124%`, variance_shift `-2.988528%`, beats random `88/96`, paired
  `0.8917`, beats matched `96/96`, leakage `False`.

Decision: diagnostic keep, not parent. This is now the strongest external
PatchTST-harm repair (`0/8` harmed, external PatchTST `-0.366358%`) and it
beats matched smoothing on every stress config, but it trails
`s14_smoothing_cap_selective_router` on clean mean and stress mean. The next
candidate should make the stable veto conditional, so it only applies to
external-harm-like low-roughness windows and leaves smoothing-cap behavior
unchanged on stress/high-risk windows.

## Conditional Stable Smoothing Cap Router

`s14_conditional_stable_cap_router` keeps the smoothing confidence cap but makes
the stable-forecast veto narrower: raw smoothing falls back to
`boundary_then_selective_median` only when the prediction is stable relative to
context and the validation-calibrated unsupported-noise score is low.

Completed evidence so far:

- L0 smoke completed `4/4`: MSE delta `-2.693919%`, DLinear `-4.912836%`,
  PatchTST `-0.475001%`, beats random `4/4`, paired `0.95`, beats matched
  `4/4`, leakage `False`.
- L1 clean_full completed `16/16`: MSE delta `-2.181291%`, MAE delta
  `-1.466981%`, DLinear `-3.753660%`, PatchTST `-0.608921%`, beats random
  `16/16`, paired `0.95`, beats matched `16/16`, leakage `False`.
- L3 external fixture completed `16/16`: mean MSE delta `-1.469623%`, DLinear
  `-2.767756%`, PatchTST `-0.171489%`, PatchTST harmed `2/8`, beats random
  `7/16`, paired `0.4375`, beats matched `13/16`, dominant action `0.6875`,
  leakage `False`.
- L2 stress completed `96/96`: mean MSE delta `-2.504963%`,
  boundary_discontinuity `-2.476395%`, high_frequency_perturbation
  `-2.894392%`, trend_drift `-2.269657%`, slope_break `-2.201136%`,
  delayed_level_shift `-2.148803%`, variance_shift `-3.039397%`, beats random
  `89/96`, paired `0.9104`, beats matched `96/96`, max MSE harm
  `-0.374383%`, leakage `False`.

Decision: diagnostic keep, not parent. The conditional veto nearly preserves
the clean/stress parent on stress and improves matched-smoothing separation, but
it still leaves external PatchTST harm at `2/8` and does not beat
`s14_smoothing_cap_selective_router` on clean/stress mean. The current frontier
therefore remains split: `s14_smoothing_cap_selective_router` is the clean/stress
parent, while `s14_stable_smoothing_cap_router` is the best external PatchTST
harm diagnostic.
