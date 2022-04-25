# How to start

Create .env with 
```
GITHUB_PAT=yourPATfromGITHUBaccount
```

## List all ETA issues

```
python prtime.py --list

Started at [2022-03-21 23:25:07.034820]
c-image-to-text
Total PR count: [262]
Total PR count with time: [17]
[created: 2022-03-18 12:53:44 -> closed:                None] assigned to [     ku-bo] state [    open] merged [False] merged by [   unknown]
...
```


## Number of customer ETA per issue

```
python prtime.py --eta-cust=949,960,939,944,943,956
python prtime.py --eta-cust=944,949,960,943,939,942,968,957,958 --state=closed
```

## Dev hours per issue

```
python prtime.py --hours=949,960,939,944,943,956
```

## Validate calculations

```
python prtime.py --validate
python prtime.py --validate --state=closed
python prtime.py --validate --state=closed --sort=merged_at
```

# Other projects

```
python prtime.py --validate --state=closed --settings=___settings.dq.json
```