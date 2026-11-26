"""Assemble app.html = app_template.html with the JSON from app_data.json inlined at __DATA__."""
import json
tpl = open('app_template.html').read(); data = open('app_data.json').read()
assert '__DATA__' in tpl, 'placeholder missing'
open('app.html', 'w').write(tpl.replace('__DATA__', data))
import os; print('app.html', os.path.getsize('app.html'), 'bytes')
