# Create weekly report

1. Fix ETAs, chide someone
```
set TRACKER_LINK=https://***.atlassian.net/browse/
set CUSTOMER=xxxcustomer
set GITHUB_PAT=XXX
python prtime.py --validate --state=closed --check-last=12w
```

2. Generate MD report
```
python prtime.py --hours --state=closed --check-last=12w
```




###
1. 15 minutes, fixed 2 code issues
