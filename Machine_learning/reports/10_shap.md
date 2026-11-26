# 10 — What the model relies on (TreeSHAP, final model)

Shares of mean |SHAP| (raw log-odds; additivity checked: contributions + base = margin on every row).
**test** = 136 unseen users (main). **dev** = the training rows (in-sample, stability check only).

## By feature family
| index | test: all rows | test: y_10 = 1 rows | dev (in-sample): all rows |
|---|---|---|---|
| Accumulation & shift (all channels) | 0.3978 | 0.5498 | 0.3888 |
| Email | 0.2358 | 0.2000 | 0.2347 |
| HTTP / web | 0.1263 | 0.0825 | 0.1272 |
| USB / file | 0.0878 | 0.0626 | 0.0968 |
| Rotation: deviation-based (nr_*) | 0.0826 | 0.0646 | 0.0841 |
| Rotation/multi-channel: team (switches, multi-channel days) | 0.0505 | 0.0305 | 0.0496 |
| Rotation: gap-alt (nr_gap_alt_2) | 0.0134 | 0.0063 | 0.0131 |
| Logon | 0.0059 | 0.0037 | 0.0057 |

## By horizon (test)
| index | all rows | y_10 = 1 rows | benign-user rows |
|---|---|---|---|
| 10 workdays | 0.3203 | 0.3920 | 0.3061 |
| 20 workdays | 0.2864 | 0.2991 | 0.2783 |
| 30 workdays | 0.0897 | 0.0641 | 0.0931 |
| 5 workdays | 0.0724 | 0.0652 | 0.0756 |
| daily | 0.2312 | 0.1797 | 0.2469 |

## Family share on attack-window rows (y_10 = 1) by switching level (test)
| index | Low | Medium | High |
|---|---|---|---|
| Accumulation & shift (all channels) | 0.5355 | 0.5695 | 0.5411 |
| Email | 0.2252 | 0.1795 | 0.1949 |
| HTTP / web | 0.0847 | 0.0763 | 0.0885 |
| Logon | 0.0031 | 0.0039 | 0.0042 |
| Rotation/multi-channel: team (switches, multi-channel days) | 0.0299 | 0.0287 | 0.0340 |
| Rotation: deviation-based (nr_*) | 0.0601 | 0.0660 | 0.0688 |
| Rotation: gap-alt (nr_gap_alt_2) | 0.0051 | 0.0072 | 0.0065 |
| USB / file | 0.0564 | 0.0689 | 0.0619 |

## Top 15 features (test)
| index | mean_abs_shap | share | family | direction |
|---|---|---|---|---|
| cumulative_deviation_10wd | 1.1048 | 0.1812 | Accumulation & shift (all channels) | higher -> more risk |
| cumulative_deviation_20wd | 0.4736 | 0.0777 | Accumulation & shift (all channels) | higher -> more risk |
| email_event_count | 0.4489 | 0.0736 | Email | higher -> more risk |
| http_event_count | 0.3500 | 0.0574 | HTTP / web | higher -> more risk |
| usb_active_days_20wd | 0.1906 | 0.0313 | USB / file | higher -> less risk |
| email_active_days_20wd | 0.1770 | 0.0290 | Email | higher -> less risk |
| external_email_count_20wd | 0.1741 | 0.0286 | Email | higher -> more risk |
| consecutive_elevated_days_20wd | 0.1717 | 0.0282 | Accumulation & shift (all channels) | higher -> more risk |
| multi_channel_days_10wd | 0.1541 | 0.0253 | Rotation/multi-channel: team (switches, multi-channel days) | higher -> less risk |
| cumulative_deviation_5wd | 0.1397 | 0.0229 | Accumulation & shift (all channels) | higher -> more risk |
| email_active_days_10wd | 0.1258 | 0.0206 | Email | higher -> less risk |
| www_visit_count | 0.1170 | 0.0192 | HTTP / web | higher -> more risk |
| combined_channel_intensity_30wd | 0.1060 | 0.0174 | Accumulation & shift (all channels) | higher -> more risk |
| external_email_count_10wd | 0.0909 | 0.0149 | Email | higher -> more risk |
| nr_handoff_10wd | 0.0861 | 0.0141 | Rotation: deviation-based (nr_*) | higher -> more risk |

## Example `shap_explanation_jsonb` (highest-risk positive test row)
```json
{
  "model_version": "mubsir-lgbm-y10-2.0.0-dev",
  "risk_score": 99.96,
  "base_value_logit": -3.9995,
  "units": "calibrated log-odds contribution (a * TreeSHAP); positive = raises risk",
  "top_features": [
    {
      "feature": "cumulative_deviation_10wd",
      "value": 21.66666603088379,
      "contribution": 4.8804,
      "family": "Accumulation & shift (all channels)",
      "direction": "raises risk"
    },
    {
      "feature": "cumulative_deviation_20wd",
      "value": 30.08333396911621,
      "contribution": 1.6868,
      "family": "Accumulation & shift (all channels)",
      "direction": "raises risk"
    },
    {
      "feature": "email_event_count",
      "value": 12.0,
      "contribution": 0.8638,
      "family": "Email",
      "direction": "raises risk"
    },
    {
      "feature": "http_event_count",
      "value": 105.0,
      "contribution": 0.6896,
      "family": "HTTP / web",
      "direction": "raises risk"
    },
    {
      "feature": "cumulative_deviation_5wd",
      "value": 12.333333015441895,
      "contribution": 0.4942,
      "family": "Accumulation & shift (all channels)",
      "direction": "raises risk"
    }
  ]
}
```
