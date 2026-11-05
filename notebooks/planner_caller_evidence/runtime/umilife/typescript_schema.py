"""Compact TypeScript interfaces derived solely from the installed public catalog."""
import json

def literal(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def ts_type(p):
    if 'enum' in p:return ' | '.join(literal(v) for v in p['enum'])
    t=p.get('type')
    if isinstance(t,list):return ' | '.join(ts_type({**p,'type':v}) for v in t)
    if t=='array':return '('+ts_type(p.get('items',{}))+')[]'
    if t=='object':
        fields=p.get('properties')
        if fields is None:return 'Record<string, unknown>'
        base='{ '+ '; '.join(literal(k)+('' if k in p.get('required',[]) else '?')+': '+ts_type(v) for k,v in fields.items())+' }'
        if p.get('x-one-of'):
            alt=['{ '+'; '.join(literal(k)+': '+ts_type(fields[k]) for k in group)+' }' for group in p['x-one-of']]
            return base+' & ('+' | '.join(alt)+')'
        return base
    return {'string':'string','integer':'number','number':'number','boolean':'boolean','null':'null'}.get(t,'unknown')

def metadata(p,path='',detailed=False):
    result=[]
    for k,v in p.items():
        if k in ('properties','items','required','type','enum','x-one-of'):continue
        if k=='description' and not detailed:continue
        result.append((path+'.' if path else '')+k+'='+literal(v))
    if p.get('type')=='integer':result.append(path+':integer')
    for k,v in p.get('properties',{}).items():result.extend(metadata(v,(path+'.' if path else '')+k,detailed))
    if p.get('items'):result.extend(metadata(p['items'],path+'[]',detailed))
    return result

def typescript_catalog(specs,detailed=False):
    lines=['interface InstalledTools {']
    for name,s in sorted(specs.items()):
        comment='; '.join([s.get('kind',''),s.get('description',''),*metadata(s['parameters'],detailed=detailed)])
        if s.get('returns') and detailed:comment+='; returns='+str(s['returns'])
        comment=comment.replace('*/','* /').replace('\n',' ')
        lines.append('  /** '+comment+' */')
        lines.append('  '+literal(name)+'(args: '+ts_type(s['parameters'])+'): unknown;')
    return '\n'.join(lines+['}'])
