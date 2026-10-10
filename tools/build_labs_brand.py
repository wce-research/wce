"""Reproduce WCE Labs exports from the existing three-part invariant symbol."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'app/static/brand'
PATHS=('M8 8h32v8H16v24H8Z','M56 56H24v-8h24V24h8Z','M28 24h8v16h-8Z')
POLYGONS=(((8,8),(40,8),(40,16),(16,16),(16,40),(8,40)),((56,56),(24,56),(24,48),(48,48),(48,24),(56,24)),((28,24),(36,24),(36,40),(28,40)))
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')

def mark(draw,x,y,size,color):
    for polygon in POLYGONS:draw.polygon([(x+a*size/64,y+b*size/64) for a,b in polygon],fill=color)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    for mode,background,ink in [('light','#f4f2ec','#20211f'),('dark','#161817','#efeee8'),('mono','none','currentColor')]:
        shapes=''.join(f'<path d="{p}"/>' for p in PATHS)
        avatar=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><title>WCE Labs — invariant</title><rect width="512" height="512" fill="{background}"/><g transform="translate(96 96) scale(5)" fill="{ink}">{shapes}</g></svg>'
        (OUT/f'wce-labs-avatar-{mode}.svg').write_text(avatar+'\n')
        logo=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 72"><title>WCE Labs</title><g transform="translate(4 4)" fill="{ink}">{shapes}</g><text x="90" y="49" fill="{ink}" font-family="Arial,Helvetica,sans-serif" font-size="40" font-weight="600" letter-spacing="-1">WCE</text><text x="202" y="45" fill="{ink}" font-family="Arial,Helvetica,sans-serif" font-size="21" letter-spacing="3">LABS</text></svg>'
        (OUT/f'wce-labs-logo-{mode}.svg').write_text(logo+'\n')
        if mode!='mono':
            image=Image.new('RGB',(1024,1024),background);mark(ImageDraw.Draw(image),192,192,640,ink)
            image.resize((512,512),Image.Resampling.LANCZOS).save(OUT/f'wce-labs-avatar-{mode}.png')
    image=Image.new('RGB',(1200,630),'#f4f2ec');draw=ImageDraw.Draw(image)
    mark(draw,80,80,144,'#20211f')
    draw.text((265,105),'WCE Labs',fill='#20211f',font=ImageFont.truetype(str(FONT),70))
    draw.line((80,280,1120,280),fill='#d3d2cb',width=2)
    draw.text((80,330),'World Configuration Engine',fill='#20211f',font=ImageFont.truetype(str(FONT),46))
    draw.text((80,416),'Compact · compositional · verifiable AI research',fill='#62635e',font=ImageFont.truetype(str(FONT),28))
    draw.text((80,548),'wcelabs.com',fill='#343cf2',font=ImageFont.truetype(str(FONT),25))
    image.save(OUT/'wce-labs-og.png')
    favicon=Image.new('RGBA',(256,256),'#f4f2ec');mark(ImageDraw.Draw(favicon),16,16,224,'#20211f')
    favicon.save(ROOT/'app/static/favicon.ico',sizes=[(16,16),(32,32),(48,48)])
    favicon.resize((180,180),Image.Resampling.LANCZOS).save(ROOT/'app/static/apple-touch-icon.png')
    (ROOT/'app/static/favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><title>WCE Labs</title><style>path{fill:#20211f}@media(prefers-color-scheme:dark){path{fill:#efeee8}}</style>'+''.join(f'<path d="{p}"/>' for p in PATHS)+'</svg>\n')
    manifest={'name':'WCE Labs','short_name':'WCE Labs','description':'World Configuration Engine — experimental architecture research','id':'/','start_url':'/','scope':'/','display':'browser','background_color':'#f4f2ec','theme_color':'#f4f2ec','icons':[{'src':'/brand/wce-labs-avatar-light.png','sizes':'512x512','type':'image/png','purpose':'any'}]}
    (ROOT/'app/static/site.webmanifest').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':build()
