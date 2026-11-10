# ACT in-box release diagnosis — 2026-10-29

## Scope and unchanged outcome

This is retrospective diagnosis of the [completed ACT baseline](act-baseline-validation-20261029.md), not retraining, parameter deployment or a revised success count. Original policy evaluation remains 0/16 successful tasks. Pi0 serving was not switched for these diagnostics. Contact analysis and sampled primary review are complete, with the original policy outcome unchanged.

## Labels, replay and fixed-state predictions

Evidence is retained under [the diagnostic directory](../outputs/act_release_diagnosis_20261029/). All sixteen complete action replays reproduced recorded pre-action states with maximum error zero. All twenty-two training episodes contain 35 release frames, with gripper action rising to approximately 0.97472: release labels are present.

The [prediction audit](../outputs/act_release_diagnosis_20261029/predictions.json) binds 134 fixed states (110 teacher, 24 learner) to final ACT update20000 and its checkpoint hash. Learner first-25 predictions match original logs exactly. None of the 24 learner queries predicts opening ≥0.8 within the first25 actions, while 22 predict that opening somewhere in the later75. For each of 22 teacher release+25 and release+50 states, the first25 predictions reach ≥0.8; at release0 this occurs in 17/22. Thus the policy is not uniformly incapable of predicting opening. Teacher/learner state differences and execution horizon are diagnostic hypotheses, not an established sole cause.

The new state-query upload was initially rejected by automatic approval review, then executed after the user's explicit authorization to use the same ACT cloud environment. No private connection details or model-state arrays are reproduced here.

## Corrected terminal branches and invalid preliminary branch

Only [corrected branches](../outputs/act_release_diagnosis_20261029/corrected_branches.json) support terminal-state intervention claims. Six unique final states were compared with holding versus opening only, each from fresh environments with full-prefix replay, identical starting integration hashes and zero recorded-state error. Both arms had 0/6 complete task success. One opening branch briefly satisfied released; the others retained contact. Opening alone is not proven sufficient to finish these terminal states.

The original `diagnosis.json` branch experiment did not restore cached dynamics after `mj_setState`. Additional instrumented replay exposed state error; [branch validity](../outputs/act_release_diagnosis_20261029/branch_validity.json) marks those preliminary branches invalid and preserves their sources/logs/arrays. Its sixteen-prefix replays and prediction queries remain independently valid. Invalid branches must not be combined with corrected results.

## Fixed middle-state suffix experiment

The [suffix plan](../outputs/act_release_diagnosis_20261029/suffix_plan.json) and [results](../outputs/act_release_diagnosis_20261029/suffix_results.json) compare six fixed intermediate states over 100 actions: original recorded replanning versus the complete100-action ACT prediction from the same observation, without forced opening. Full-chunk execution reached released at least once in 6/6 versus 1/6 for the baseline, often transiently. Both had 0/6 complete task success within the two-second window. Full-chunk terminal flags were inside/released/supported/stable in 5/6; the remaining case lacked stable.

These short diagnostic suffixes are not additional sixteen-case policy successes or deployment evidence. No `n_action_steps=100` setting was deployed, and no model improvement or causal guarantee follows. The completed contact and visual evidence below qualifies these observations; no execution-horizon change follows.


## Instrumented contact and physical opening

The [instrumented contacts](../outputs/act_release_diagnosis_20261029/contacts_instrumented.json) cover all six corrected cases with strict prefix and branch-state error zero. Every case had positive finger-to-box-bottom contact force; case6 additionally had gripper-bar/box-back contact. After an opening target of 1 (projected to 0.97472), actual jaw opening reached only approximately 0.546–0.657 and finger/banana contact remained. Commanded opening and physical release are therefore distinct. These local measurements do not establish box contact as the unique cause of all failures.

The [29-check audit](../outputs/act_release_diagnosis_20261029/audit.json) and [completion receipt](../outputs/act_release_diagnosis_20261029/completion.json) bind the completed diagnosis. [Primary review](../outputs/act_release_diagnosis_20261029/primary_review.json) records inspection of [all six prediction horizons](../outputs/act_release_diagnosis_20261029/gripper_horizons.png) and [all six corrected hold/open comparisons](../outputs/act_release_diagnosis_20261029/contact_branches_sheet.png). This is static image review, not complete video playback. The invalid cached-dynamics preliminary branches remain preserved and excluded from conclusions.

The user explicitly authorized upload of the 134 simulation samples before successful execution. There was no new training, deployed parameter change or service switch. Original ACT weights and the 0/16 policy result remain unchanged. Short suffix release flags motivate further investigation, not a claim of successful task repair.
