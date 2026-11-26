# 12 — Release verification

**24/24 checks PASS.**

## Bundle (`model/`)
```
bed7670c6664186b05366d728634638cb7a1d4ea7977f648c09bb58ff70c45ce  lgbm_y10.txt
59ccbd5ff92111a0311cb5588eef5a91b58c975f106c3d4d96f4e8eb7708a84c  manifest.json
d4b21d44ae04f4547b385ad32b046e3f8b0c274eab8f361a99c2f47f28968048  mubsir_infer.py
4f8e716b00cfe29d8686a71d17750152e8fc4b3ff6aebe5b417c67ac3f5be3cb  verify_bundle.py
51ce3d94e18775a19c9e1a2097f06b8757d207d12ccde0ea19884164df0ae610  example_input.parquet
0f50a3c2a2f2050712cda6cd6435ab5d7689e073b90b30ddf734467829f0c202  requirements.txt
977b24e72251458c4a169fb4e2ea0b3a5edaa19824c075d613089e69b23baea6  README.md
705fd4d6451a31d36b3df7de96f83f30ac976c9b4a6d1e51671d8e2f33e2d0da  .gitattributes
```
Built with: lightgbm==4.7.0, numpy==2.5.5, pandas==3.1.1, pyarrow==26.0.1.

## Checks
- [PASS] verify_bundle.py (stdlib) passes 8/8 files OK
- [PASS] model file unchanged since freeze 
- [PASS] manifest unchanged since freeze 
- [PASS] split / rules / dataset unchanged since freeze 
- [PASS] scorer feature order = pipeline feature list (100) 
- [PASS] bundle scorer reproduces the frozen test risk exactly max diff 0.0e+00
- [PASS] levels follow the frozen operators 
- [PASS] score_users: risk_peak = max daily risk per user 
- [PASS] explain: base + all contributions = calibrated logit (risk reproduced) 
- [PASS] explain: one dict per row with top features 
- [PASS] missing feature column raises ValueError 
- [PASS] NaN outside channel_distribution_shift_* raises ValueError 
- [PASS] NaN in channel_distribution_shift_* accepted 
- [PASS] infinite value raises ValueError 
- [PASS] Stage A decision: lightgbm, y_10 
- [PASS] Stage B tuning not adopted 
- [PASS] no ablation adopted 
- [PASS] pre-test audit all PASS before test 24 checks
- [PASS] manifest params = defaults, 237 trees 
- [PASS] no test user in CV folds 
- [PASS] dev/test schedule groups disjoint 
- [PASS] no forbidden column among model features 
- [PASS] final model trained on dev rows only 
- [PASS] all stage reports present []
