# ESPN's hidden API endpoints

## College Football 
**Latest News**: 
**Latest Scores**: http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard
    - query params:
        - calendar: 'blacklist'
        - dates: any date in YYYYMMDD
**Team Information**: http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/:team
    - params: 
        - team: some team abbreviation (EX: 'all' for Allegheny, 'gt' for Georgia Tech, 'wisconsin' for Wisconsin)

Will update with more information as I find more...