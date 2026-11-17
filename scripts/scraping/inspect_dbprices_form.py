from bs4 import BeautifulSoup

soup = BeautifulSoup(open('data/metadata/dca_dbprices_sample.html', encoding='utf-8').read(), 'html.parser')
for f in soup.find_all('form'):
    print("Form action:", f.get('action'))
    for inp in f.find_all(['input', 'select']):
        print(f"  {inp.get('name')}: {inp.get('type', 'select')} = {inp.get('value')}")
