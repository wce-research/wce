/* Rasterize the repo-native SVGs; no image generation or image editing service.
 * NODE_PATH may point to a temporary install of playwright-core.
 */
const {chromium}=require('playwright-core');
const fs=require('fs');const path=require('path');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.WCE_CHROMIUM,headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({deviceScaleFactor:1});
 const root=__dirname;
 const folders=fs.readdirSync(path.join(root,'finalists')).map(id=>path.join(root,'finalists',id));
 for(const dir of folders){
  for(const name of fs.readdirSync(dir).filter(x=>x.endsWith('.svg'))){
   const svg=fs.readFileSync(path.join(dir,name),'utf8');
   const width=Number(svg.match(/width="(\d+)"/)[1]);const height=Number(svg.match(/height="(\d+)"/)[1]);
   await page.setViewportSize({width,height});
   await page.setContent('<style>html,body{margin:0;padding:0;background:transparent}svg{display:block}</style>'+svg);
   await page.screenshot({path:path.join(dir,name.replace('.svg','.png')),omitBackground:true});
  }
  const svg=fs.readFileSync(path.join(dir,'favicon.svg'),'utf8');
  for(const size of [16,32,48,180,192,512]){
   await page.setViewportSize({width:size,height:size});
   await page.setContent('<style>html,body{margin:0}svg{display:block;width:100vw;height:100vh}</style>'+svg);
   await page.screenshot({path:path.join(dir,'favicon-'+size+'.png')});
  }
 }
 for(const name of ['initial-12','finalists-5']){
  const svg=fs.readFileSync(path.join(root,'boards',name+'.svg'),'utf8');
  await page.setViewportSize({width:Number(svg.match(/width="(\d+)"/)[1]),height:Number(svg.match(/height="(\d+)"/)[1])});
  await page.setContent('<style>html,body{margin:0}svg{display:block}</style>'+svg);
  await page.screenshot({path:path.join(root,'boards',name+'.png')});
 }
 await page.setViewportSize({width:1440,height:1000});
 await page.goto('file://'+path.join(root,'index.html'));
 await page.screenshot({path:path.join(root,'boards','exploration-desktop.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Review page mobile overflow');
 await page.screenshot({path:path.join(root,'boards','exploration-mobile.png'),fullPage:true});
 console.log('Rendered five complete asset kits, two contact sheets, desktop/mobile review page');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
