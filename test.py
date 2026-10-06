import requests
import re
h = requests.get("https://hakitrade-production.up.railway.app").text
m = re.search(r'src="(/assets/index-[^\"]+\.js)"', h)
if m:
    js = m.group(1)
    print("Found JS:", js)
    j = requests.get("https://hakitrade-production.up.railway.app" + js).text
    print("Has Backend URL:", "backend-production" in j)
else:
    print("No JS found")
