import requests

# -------------------------------
# Βάζεις εδώ το δικό σου API key
# -------------------------------
api_key = "227cdea05de943bf04fcab225cec1457"

# URL για να πάρουμε όλα τα πρωταθλήματα
url = "https://v3.football.api-sports.io/leagues"
headers = {"x-apisports-key": api_key}

# Κάνουμε request στο API
response = requests.get(url, headers=headers).json()
leagues = response['response']

# -------------------------------
# Λίστα με τα πρωταθλήματα που θέλουμε
# Βάζεις τα league_id που σε ενδιαφέρουν
# -------------------------------
football_leagues = [39, 197, 494, 140, 135, 145, 203, 88, 94, 61, 78]  # π.χ. Premier League, La Liga, Serie A, Bundesliga, Ligue 1

# Φτιάχνουμε λίστα μόνο με τα επιλεγμένα πρωταθλήματα
selected_leagues = []
for l in leagues:
    if l['league']['id'] in football_leagues:
        selected_leagues.append({
            "name": l['league']['name'],
            "id": l['league']['id'],
            "country": l['country']['name']
        })

# Τυπώνουμε τα επιλεγμένα πρωταθλήματα
for league in selected_leagues:
    print(league['name'], league['id'], league['country'])
