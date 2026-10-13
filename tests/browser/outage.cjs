/* Isolated outage preview: no application server, no cloud operations. */
const {chromium}=require('playwright');
const AxeBuilder=require('@axe-core/playwright').default;
const {spawn}=require('child_process');
const fs=require('fs');
const {pathToFileURL}=require('url');
const path=require('path');
const output='/tmp/wce-outage-review';
let server,browser;
const check=(value,message)=>{if(!value)throw Error(message);};
(async()=>{
 fs.mkdirSync(output,{recursive:true});
 const base=await new Promise((resolve,reject)=>{
  server=spawn('make',['--no-print-directory','outage-preview'],{detached:true,stdio:['ignore','pipe','inherit']});
  const timeout=setTimeout(()=>reject(Error('Preview startup timed out')),15000);
  server.stdout.on('data',data=>{const match=data.toString().match(/Outage preview: (http:\/\/127\.0\.0\.1:\d+)\//);if(match){clearTimeout(timeout);resolve(match[1]);}});
  server.on('error',reject);
 });
 browser=await chromium.launch({headless:true});
 let renders=0,scans=0;
 for(const javaScriptEnabled of [false,true]){
  const context=await browser.newContext({javaScriptEnabled,reducedMotion:'reduce'});
  const page=await context.newPage();
  const requests=[];page.on('request',request=>requests.push(request.url()));
  if(!javaScriptEnabled){
   for(const locale of ['fr','en']){
    await page.goto(pathToFileURL(path.resolve(`ops/outage/${locale}.html`)).href);
    check(await page.locator('h1').isVisible(),'Standalone file renders without HTTP/application server');
    check(await page.locator('header svg').isVisible(),'Standalone official branding');
   }
  }
  for(const locale of ['fr','en']){
   for(const [width,height] of [[320,568],[390,844],[768,1024],[1440,900],[2560,1440]]){
    await page.setViewportSize({width,height});requests.length=0;
    const url=base+'/'+locale+'/?preview=1';
    const response=await page.goto(url);
    check(response.status()===503,'Outage preserves 503');
    check(await page.locator('html').getAttribute('lang')===locale,'Correct document language');
    check(await page.locator('header svg').count()===1,'Official inline logo');
    check(await page.locator('main').count()===1,'Main landmark');
    check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'No horizontal overflow');
    if(width>=390)check(!await page.evaluate(()=>document.documentElement.scrollHeight>innerHeight),'No unnecessary vertical scrolling');
    check(requests.every(value=>value===url)&&requests.length===1,'Only current document request');
    await page.keyboard.press('Tab');
    check(await page.locator('.retry').evaluate(node=>node===document.activeElement),'Retry keyboard focus');
    check(await page.locator('.retry').evaluate(node=>getComputedStyle(node).outlineStyle)==='solid','Visible focus');
    await page.keyboard.press('Enter');await page.waitForLoadState('load');
    check(page.url()===url,'Retry preserves current URL including query');
    check(await page.locator('body').evaluate(node=>getComputedStyle(node).animationName)==='none','No motion');
    await page.screenshot({path:`${output}/${locale}-${width}-${javaScriptEnabled?'js':'nojs'}.png`,fullPage:true});
    if(javaScriptEnabled){
     const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
     check(result.violations.length===0,JSON.stringify(result.violations));scans++;
    }
    renders++;
   }
  }
  await context.close();
 }
 fs.writeFileSync(output+'/results.json',JSON.stringify({renders,scans,violations:0,externalRequests:0,javaScriptDisabledRenders:10},null,2));
 console.log(`${renders} responsive renders; ${scans} WCAG AA scans; no violations; JS-disabled retry verified. Screenshots: ${output}`);
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
 if(browser)await browser.close();if(server)process.kill(-server.pid,'SIGTERM');
});
