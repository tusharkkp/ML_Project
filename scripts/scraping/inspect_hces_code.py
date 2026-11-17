import requests

js = requests.get('https://esankhyiki.mospi.gov.in/static/js/main.ecf16667.js', verify=False).text
idx = js.find('case"HCES":')
if idx != -1:
    print(js[idx:idx+1500])
