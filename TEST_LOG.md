# Test log (fill in "Observed" and "Fix" while testing the running app)

| # | Test | Expected | Observed | Pass? | Fix |
|---|---|---|---|---|---|
| 1 | Open app without a Groq key | Data, prediction, group, image, comments all work | | | |
| 2 | Choose "Example P02_0004" and click Estimate | Estimate in units + group shown | (offline check: ~42 units, Group 0) | | |
| 3 | Pick a sample image | 28x28x1 input, classes top/trouser/bag | | | |
| 4 | Submit a blank comment | "Empty" message, not counted in totals | | | |
| 5 | Enter an invalid value / category | Clear message before prediction | (offline check of validation function: passed) | | |
| 6 | Negative price (-500) | Rejected: "price_pkr must be between 800 and 5000" | (offline check: passed) | | |
| 7 | Groq mode with no key / rate limit | Warning + offline summary shown, app still works | | | |
| 8 | Rerun a widget / restart app | Models load, no training | | | |
| 9 | Open repo on another computer | README is enough to run it | | | |
