# Guardian Forensics — ic32

## Holdout clean_v2 (backtest baseline)

- **all** (n=936): guardian 73.3% | early-cut losers 53 | dir_ok_12h losers 111
- **losers** (n=355): guardian 50.7% | early-cut losers 53 | dir_ok_12h losers 111
- **losers_dir_ok_12h** (n=111): guardian 46.8% | early-cut losers 24 | dir_ok_12h losers 111
- **early_guardian_losers** (n=53): guardian 100.0% | early-cut losers 53 | dir_ok_12h losers 24
- **guardian_losers** (n=180): guardian 100.0% | early-cut losers 53 | dir_ok_12h losers 52
- **sl_losers** (n=173): guardian 0.0% | early-cut losers 0 | dir_ok_12h losers 57

## Holdout continuation_v1 (production Guardian)

- **all** (n=492): guardian 72.4% | mom_exit 51.2% | hold med 7.0
- **losers** (n=136): guardian 22.1% | mom_exit 5.9% | hold med 9.5
- **losers_dir_ok_12h** (n=41): guardian 43.9% | mom_exit 9.8% | hold med 12.0
- **early_guardian_losers** (n=11): guardian 100.0% | mom_exit 0.0% | hold med 2.0
- **guardian_losers** (n=30): guardian 100.0% | mom_exit 26.7% | hold med 5.0
- **sl_losers** (n=96): guardian 0.0% | mom_exit 0.0% | hold med 12.0

## Live VPS

- Closed: 147 | Guardian exits: 103 (WR 50.5%)
- Early-cut guardian losers (hold<=3): 26
- Momentum exits: 13 vs guardian_exit only: 90
- Hold median: all 3.0 | guardian los 3.0