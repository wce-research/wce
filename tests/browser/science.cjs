/* Local-only public science audit. Tooling lives outside the production image. */
const {chromium}=require('playwright');
const AxeBuilder=require('@axe-core/playwright').default;
const {spawn}=require('child_process');
const fs=require('fs');
const path=require('path');
const root=path.resolve(__dirname,'../..');
const artifacts=process.env.WCE_BROWSER_ARTIFACTS||'/tmp/wce-frontend-audit';
const report={renders:0,axeScans:0,violations:[],timings:[],screenshots:[],checks:[]};
let server,browser;
function check(value,message){if(!value)throw Error(message);}
function start(){return new Promise((resolve,reject)=>{
 server=spawn('make',['--no-print-directory','frontend-serve'],{cwd:root,detached:true,env:{...process.env,WCE_HOST:'127.0.0.1',WCE_PORT:'0'},stdio:['ignore','pipe','pipe']});
 const timer=setTimeout(()=>reject(Error('Local test server did not start')),15000);
 server.stderr.on('data',data=>{const match=data.toString().match(/WCE listening on 127\.0\.0\.1:(\d+)/);if(match){clearTimeout(timer);resolve('http://127.0.0.1:'+match[1]);}});
 server.on('error',reject);server.on('exit',code=>{clearTimeout(timer);if(code)reject(Error('Local test server exited '+code));});
});}
async function screenshot(page,name){const file=path.join(artifacts,name+'.png');await page.screenshot({path:file,fullPage:true,animations:'disabled'});report.screenshots.push(file);}
async function navigate(page,url){
 for(let attempt=0;attempt<3;attempt++){
  try{return await page.goto(url,{waitUntil:'load'});}
  catch(error){if(!String(error).includes('net::ERR_NETWORK_CHANGED')||attempt===2)throw error;console.log('Retrying local navigation after host network change');}
 }
}
(async()=>{
 fs.mkdirSync(artifacts,{recursive:true});const base=await start();
 browser=await chromium.launch({headless:true,args:process.env.WCE_BROWSER_NO_SANDBOX==='1'?['--no-sandbox']:[]});
 const context=await browser.newContext({locale:'en-US',reducedMotion:'reduce'});
 const page=await context.newPage();const errors=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto(base+'/');
 const sitemap=await (await context.request.get(base+'/sitemap.xml')).text();
 const routes=[...sitemap.matchAll(/<loc>https:\/\/wcelabs.com([^<]+)<\/loc>/g)].map(match=>match[1]);
 check(routes.length>60,'Sitemap must include the complete bilingual route catalog');
 const newRoutes=['/research/programme','/research/deep-composition','/research/high-precision','/architecture/research-map'];
 const focused=process.env.WCE_AUDIT_SCOPE==='science';
 for(const width of [1440,768,390]){
  await page.setViewportSize({width,height:1000});
  for(const theme of ['light','dark']){
   await page.evaluate(value=>{localStorage.setItem('wce.theme',value);},theme);
   for(const route of routes.filter(route=>!focused||newRoutes.some(suffix=>route.endsWith(suffix))||route.endsWith('/'))){
    const response=await navigate(page,base+route);
    check(response.status()===200,'Route '+route);
    check(await page.locator('html').getAttribute('data-theme')===theme,'Theme '+route);
    check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Overflow '+route+' '+width+' '+theme);
    check(await page.locator('h1').count()===1,'One h1 '+route);
    check(await page.locator('main').count()===1,'Main landmark '+route);
    const locale=route.split('/')[1];check(await page.locator('html').getAttribute('lang')===locale,'Language '+route);
    const canonical=await page.locator('link[rel=canonical]').getAttribute('href');check(canonical==='https://wcelabs.com'+route,'Canonical '+route);
    const links=await page.locator('a[href]').evaluateAll(nodes=>nodes.map(node=>node.getAttribute('href')));
    for(const href of links){if(href.startsWith('/en/')||href.startsWith('/fr/'))check(routes.includes(href.split('#')[0]),'Internal link '+href);}
    const science=newRoutes.some(suffix=>route.endsWith(suffix));
    if(width===1440||(science&&width===390)){
     const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();report.axeScans++;
     report.violations.push(...result.violations.map(item=>({route,width,theme,id:item.id,impact:item.impact,nodes:item.nodes.map(node=>({target:node.target,summary:node.failureSummary}))})));
    }
    if(science&&width!==768){
     await screenshot(page,route.slice(1).replaceAll('/','-')+'-'+width+'-'+theme);
     report.timings.push({route,width,theme,...await page.evaluate(()=>{const n=performance.getEntriesByType('navigation')[0];return {domMs:n.domContentLoadedEventEnd,loadMs:n.loadEventEnd,htmlBytes:n.decodedBodySize,resourceBytes:performance.getEntriesByType('resource').reduce((sum,row)=>sum+row.decodedBodySize,0),domNodes:document.querySelectorAll('*').length};})});
    }
    report.renders++;
    if(report.renders%50===0){fs.writeFileSync(path.join(artifacts,'audit.json'),JSON.stringify(report,null,2));console.log('Audited '+report.renders+' page renders; '+report.axeScans+' accessibility scans; '+report.violations.length+' violations');}
   }
  }
 }
 for(const locale of ['en','fr']){
  await page.goto(base+'/'+locale+'/architecture/research-map');
  await page.locator('[data-decision=accept]').focus();await page.keyboard.press('Enter');
  check(await page.locator('.transaction-scene').getAttribute('class').then(value=>value.includes('is-accepted')),'Accepted state');
  await page.locator('[data-decision=reject]').focus();await page.keyboard.press('Enter');
  check(await page.locator('.transaction-scene').getAttribute('class').then(value=>!value.includes('is-accepted')),'Rollback preserves reference');
  check(await page.locator('[data-transaction-result]').innerText().then(value=>value.includes('A')),'Rollback status announcement');
  check(await page.locator('.transaction-point').evaluate(node=>getComputedStyle(node).transitionDuration)==='0s','Reduced motion transaction');
  await page.emulateMedia({reducedMotion:'no-preference'});
  check(await page.locator('.transaction-point').evaluate(node=>getComputedStyle(node).transitionDuration)!=='0s','Informative transition enabled');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.goto(base+'/'+locale+'/research/programme');
  const first=page.locator('.science-timeline>li>details>summary').first();await first.focus();await page.keyboard.press('Enter');
  check(await page.locator('.science-timeline>li>details').first().getAttribute('open')!==null,'Keyboard timeline');
 }
 report.checks.push('Keyboard acceptance/rollback, live status, reduced motion, keyboard milestone disclosure');
 const french=await browser.newContext({locale:'fr-FR'});const fp=await french.newPage();await fp.goto(base+'/');
 check(await fp.locator('html').getAttribute('lang')==='fr','Browser language detection');
 check(await fp.locator('.hero .lead').innerText().then(value=>value.includes('programme de recherche ouvert')),'French home explanation');
 await fp.locator('[data-language=en]').click();await fp.goto(base+'/');
 check(fp.url().endsWith('/en/'),'Saved English selection');await french.close();
 const english=await browser.newContext({locale:'en-US'});const ep=await english.newPage();await ep.goto(base+'/');
 check(await ep.locator('html').getAttribute('lang')==='en','English default');
 check(await ep.locator('.hero .lead').innerText().then(value=>value.includes('open research programme')),'English home explanation');
 await ep.locator('[data-language=fr]').click();await ep.goto(base+'/research');
 check(await ep.locator('html').getAttribute('lang')==='fr','Saved French selection on unprefixed page');await english.close();
 const nojs=await browser.newContext({javaScriptEnabled:false});const np=await nojs.newPage();
 for(const route of newRoutes){await np.goto(base+'/en'+route);check(await np.locator('h1').count()===1,'No-JS science route');}
 await np.goto(base+'/en/architecture/research-map');check(await np.locator('.science-flow').count()>=3,'No-JS diagrams');await nojs.close();
 report.checks.push('Browser language detection, explicit selection persistence, no-JavaScript science views');
 // Run the established regression suite against the same temporary local server.
 await new Promise((resolve,reject)=>{const child=spawn('make',['--no-print-directory','frontend-browser'],{cwd:root,env:{...process.env,WCE_BASE_URL:base},stdio:'inherit'});child.on('exit',code=>code?reject(Error('Existing browser suite failed '+code)):resolve());child.on('error',reject);});
 check(errors.length===0,'JavaScript errors: '+errors.join('; '));
 fs.writeFileSync(path.join(artifacts,'audit.json'),JSON.stringify(report,null,2));
 check(report.violations.length===0,'Accessibility violations: '+report.violations.length+'; inspect audit.json');
 console.log('PASS: '+report.renders+' FR/EN responsive/theme renders; '+report.axeScans+' WCAG AA scans; evidence diagrams, language persistence, reduced motion, no JS and existing browser suite');
})().catch(error=>{fs.mkdirSync(artifacts,{recursive:true});fs.writeFileSync(path.join(artifacts,'audit.json'),JSON.stringify(report,null,2));console.error(error);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(server){try{process.kill(-server.pid,'SIGTERM');}catch(error){if(error.code!=='ESRCH')throw error;}}});
