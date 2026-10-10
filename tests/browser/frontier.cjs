/* UI regression checks. Playwright is a test tool, never a web runtime dependency. */
const {chromium}=require('playwright-core');
const {spawn}=require('child_process');
const readline=require('readline');
const fs=require('fs');
const path=require('path');
const sourceRoot=path.resolve(__dirname,'../..');
const origin=process.env.WCE_BASE_URL||'http://127.0.0.1:3018';
const fixture=process.env.WCE_RENDER_FIXTURES;
const sshTarget=process.env.WCE_SSH_TARGET;
const screenshotDirectory=process.env.WCE_BROWSER_ARTIFACTS;
let gateway,browser;
const jobs=new Map();let serial=0;
const quote=text=>"'"+text.replace(/'/g,"'\\''")+"'";
const py=`import sys,json,base64,urllib.request,urllib.error
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  return None
opener=urllib.request.build_opener(NoRedirect)
for line in sys.stdin:
 q={}
 try:
  q=json.loads(line)
  body=base64.b64decode(q['body']) if q.get('body') else None
  req=urllib.request.Request(${JSON.stringify(origin)}+q['path'],data=body,headers=q.get('headers',{}),method=q['method'])
  try:res=opener.open(req,timeout=10)
  except urllib.error.HTTPError as error:res=error
  with res:answer={'id':q['id'],'status':res.status,'headers':dict(res.headers),'body':base64.b64encode(res.read()).decode()}
  print(json.dumps(answer),flush=True)
 except Exception as error:print(json.dumps({'id':q.get('id'),'error':str(error)}),flush=True)
`;
function request(method,url,headers={},body){
 return new Promise((resolve,reject)=>{const id=++serial;jobs.set(id,{resolve,reject});gateway.stdin.write(JSON.stringify({id,method,path:url,headers,body:body?body.toString('base64'):null})+'\n');});
}
function check(condition,message){if(!condition)throw Error(message);}
async function screenshot(page,name){if(screenshotDirectory){fs.mkdirSync(screenshotDirectory,{recursive:true});await page.screenshot({path:path.join(screenshotDirectory,name+'.png'),animations:'disabled'});}}
(async()=>{
 if(sshTarget){
  const args=['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes'];
  if(process.env.WCE_SSH_IDENTITY)args.push('-o','IdentitiesOnly=yes','-i',process.env.WCE_SSH_IDENTITY);
  args.push(sshTarget,'python3 -u -c '+quote(py));
  gateway=spawn('ssh',args);gateway.stderr.on('data',data=>process.stderr.write(data));
  readline.createInterface({input:gateway.stdout}).on('line',line=>{const result=JSON.parse(line);const job=jobs.get(result.id);jobs.delete(result.id);if(result.error)job.reject(Error(result.error));else job.resolve(result);});
  gateway.on('exit',code=>{for(const job of jobs.values())job.reject(Error('SSH relay exited '+code));});
 }
 browser=await chromium.launch({headless:true,...(process.env.WCE_CHROMIUM?{executablePath:process.env.WCE_CHROMIUM}:{}),args:process.env.WCE_BROWSER_NO_SANDBOX==='1'?['--no-sandbox']:[]});
 const page=await browser.newPage({colorScheme:'light'});const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const base=(fixture||sshTarget)?'http://wce.test':origin;
 if(fixture){
  const routes=JSON.parse(fs.readFileSync(path.join(fixture,'routes.json')));
  await page.route(base+'/**',async route=>{
   const url=new URL(route.request().url());
   if(routes[url.pathname])return route.fulfill({body:fs.readFileSync(path.join(fixture,routes[url.pathname])),contentType:'text/html'});
   const name=url.pathname.slice(1);
   const assets=['styles.css','theme.js','site.js','frontier.js','app.js','api.js','favicon.svg','favicon.ico','apple-touch-icon.png'];
   if(assets.includes(name))return route.fulfill({body:fs.readFileSync(path.join(sourceRoot,'app/static',name)),contentType:name.endsWith('.js')?'text/javascript':name.endsWith('.css')?'text/css':name.endsWith('.svg')?'image/svg+xml':'application/octet-stream'});
   return route.fulfill({status:404,body:'Not found'});
  });
 }else if(sshTarget){
  await page.route(base+'/**',async route=>{const req=route.request();const url=new URL(req.url());const headers=await req.allHeaders();headers.host=url.host;const res=await request(req.method(),url.pathname+url.search,headers,req.postDataBuffer());await route.fulfill({status:res.status,headers:res.headers,body:Buffer.from(res.body,'base64')});});
 }
 async function verifyLaboratory(){
 for(const locale of ['fr','en']){
  await page.setViewportSize({width:390,height:844});await page.goto(base+'/'+locale+'/lab');
  for(const theme of ['light','dark']){
   await page.evaluate(value=>{document.documentElement.dataset.theme=value;},theme);
   const contrast=await page.locator('.console label').evaluate(el=>({text:getComputedStyle(el).color,surface:getComputedStyle(el.closest('.console')).backgroundColor}));
   check(contrast.text==='rgb(239, 238, 232)','Readable console label '+theme);
  }
  const command=page.locator('#command');
  await page.locator('[data-command]').first().click();
  check((await command.inputValue()).includes(locale==='fr'?'Soulève':'Lift'),'Example localisation');
  await page.locator('#send').click();await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  check(await page.locator('#history .entry').first().innerText().then(t=>t.includes('OK.')),'Engine action');
  await page.locator('[data-command]').nth(1).click();await page.locator('#send').click();
  await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  check(await page.locator('#history .entry').first().innerText().then(t=>t.includes('2')),'Engine observation');
  check(await page.locator('#history .entry span').first().evaluate(el=>getComputedStyle(el).color)==='rgb(239, 238, 232)','Readable engine answer');
  const rejected=locale==='fr'?'Avance le temps de 61 secondes':'Advance time by 61 seconds';
  await command.fill(rejected);await page.locator('#send').click();await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  check(await page.locator('#history .entry').first().getAttribute('class').then(t=>t.includes('error')),'Server validation');
  check(await command.inputValue()===rejected,'Preserve rejected command');
  const fail=route=>route.abort('failed');await page.route(base+'/api/command',fail);
  await page.locator('#send').click();await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  check(await command.inputValue()===rejected,'Preserve network failure draft');
  check(await page.locator('#history .entry').first().innerText().then(t=>t.includes(locale==='fr'?'inaccessible':'could not be reached')),'Friendly network error');
  await page.unroute(base+'/api/command',fail);
  check(await command.isEnabled(),'Retry enabled');
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Lab mobile overflow');
  await page.locator('.console').scrollIntoViewIfNeeded();await screenshot(page,'lab-feedback-'+locale);
 }
 }
 if(process.env.WCE_BROWSER_SCOPE==='footer'){
  for(const locale of ['fr','en'])for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});await page.goto(base+'/'+locale+'/');
   await page.locator('footer').scrollIntoViewIfNeeded();
   check(await page.locator('.footer-directory>div').count()===5,'Footer sections');
   check(await page.locator('.footer-directory').innerText().then(t=>t.includes(locale==='fr'?'Recherche ouverte':'Open research')),'Footer language');
   check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Footer overflow');
   if(screenshotDirectory){fs.mkdirSync(screenshotDirectory,{recursive:true});await page.locator('footer').screenshot({path:path.join(screenshotDirectory,'footer-'+locale+'-'+width+'.png'),animations:'disabled'});}
  }
  check(errors.length===0,'Footer JS errors');console.log('PASS: footer FR/EN desktop/mobile; five sections; localised titles; no overflow or JS errors');return;
 }
 if(process.env.WCE_BROWSER_SCOPE==='home'){
  for(const locale of ['fr','en'])for(const width of [1440,390])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1000});await page.goto(base+'/'+locale+'/');
   await page.evaluate(value=>{document.documentElement.dataset.theme=value;},theme);
   check(await page.locator('.study-card').count()===3,'Three scientific records');
   check(await page.locator('.study-value').count()===3,'Source counts');
   check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Home overflow');
   await screenshot(page,'home-'+locale+'-'+width+'-'+theme);
   if(screenshotDirectory)await page.locator('.research-preview').screenshot({path:path.join(screenshotDirectory,'studies-'+locale+'-'+width+'-'+theme+'.png'),animations:'disabled'});
  }
  check(errors.length===0,'Home JS errors');console.log('PASS: 8 home renders; FR/EN; desktop/mobile; light/dark; three scientific records; no overflow or JS errors');return;
 }
 if(process.env.WCE_BROWSER_SCOPE==='lab'){
  await verifyLaboratory();check(errors.length===0,'Lab JS errors');
  console.log('PASS: laboratory FR/EN; light/dark contrast; real engine actions; server validation; preserved network-error draft; no overflow or JS errors');return;
 }
 const routes=['/','/research','/vision','/open','/open/source','/open/contribute','/lab','/architecture','/architecture/measurement','/research/measurement','/research/continual','/research/discovery'];
 let renders=0;
 for(const width of [1440,768,390]){
  await page.setViewportSize({width,height:width===390?844:1000});
  for(const locale of ['fr','en'])for(const route of routes){
   const response=await page.goto(base+'/'+locale+(route==='/'?'/':route),{waitUntil:'networkidle'});
   check(response.status()===200,'HTTP '+route);check(await page.getAttribute('html','lang')===locale,'Locale '+route);
   check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Overflow '+locale+route+' '+width+' '+JSON.stringify(await page.evaluate(()=>Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>[e.tagName,e.className,Math.round(e.getBoundingClientRect().right)]).slice(0,12))));renders++;
  }
 }
 await page.setViewportSize({width:1440,height:1000});
 await page.goto(base+'/fr/');
 const anchor=await page.locator('.configuration-anchor').innerHTML();
 const initial=await page.locator('[data-primitive="0"]').getAttribute('transform');
 await page.locator('[data-configuration=rotation]').click();
 check(await page.locator('#configuration-state').innerText()==='B','Rotation state');
 check(await page.locator('[data-primitive="0"]').getAttribute('transform')!==initial,'Rotation geometry');
 await page.locator('[data-configuration=composition]').click();
 check(await page.locator('#configuration-operator').innerText()==='T ∘ R','Composition operator');
 check(await page.locator('.configuration-anchor').innerHTML()===anchor,'Fixed reference changed');
 check(await page.locator('[data-primitive]').count()===3,'Primitive count changed');
 check(await page.locator('[data-configuration][aria-pressed=true]').count()===1,'Selection state');
 await screenshot(page,'configuration-day');
 await page.locator('.theme-toggle').click();await page.waitForTimeout(220);await screenshot(page,'configuration-night');
 await page.locator('[data-language=en]').click();
 check(await page.locator('#configuration-description').innerText().then(text=>text.includes('Three independent')),'English study');
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:width===390?844:1000});
  await page.goto(base+'/en/research/continual');
  const metricKeys=await page.locator('#evidence-metric option').evaluateAll(nodes=>nodes.map(node=>node.value));
  check(metricKeys.length===9,'Metric count');
  for(const metric of metricKeys){
   await page.locator('#evidence-metric').selectOption(metric);
   const panel=page.locator('[data-metric-panel="'+metric+'"]');
   check(await page.locator('.metric-panel:visible').count()===1,'Metric selection');
   check(await panel.locator('.estimate-row').count()===9,'Method count');
   check(await panel.locator('.metric-guide').isVisible(),'Metric guidance');
   await panel.locator('.evidence-table>summary').click();
   check(await panel.locator('table').isVisible(),'Exact values table');
   check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Metric table overflow '+metric);
  }
  await page.locator('#evidence-metric').selectOption('retention');
  await page.locator('.evidence-record').scrollIntoViewIfNeeded();await screenshot(page,'continuum-'+width);
  await page.goto(base+'/en/research/discovery');
  check(await page.locator('.hypothesis-card').count()===10,'Hypothesis count');
  for(const [key,n] of [['supported',4],['refuted',3],['inconclusive',3],['all',10]]){await page.locator('[data-verdict-filter='+key+']').click();check(await page.locator('.hypothesis-card:visible').count()===n,'Verdict filter '+key);}
  check(await page.locator('.hypothesis-title').count()===10,'Readable hypothesis titles');
  for(const hypothesis of ['H2','H5','H10']){await page.locator('[data-hypothesis='+hypothesis+']>summary').click();check(await page.locator('[data-hypothesis='+hypothesis+'] pre').isVisible(),'Hypothesis fields');}
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Hypothesis overflow');
  await page.locator('.evidence-record').scrollIntoViewIfNeeded();await screenshot(page,'verdicts-'+width);
  await page.goto(base+'/fr/research/measurement');
  check(await page.locator('.coverage-row').count()===5,'Source coverage');
  const fingerprint=page.locator('.artifact-hash>summary');await fingerprint.focus();await page.keyboard.press('Enter');
  check(await page.locator('.artifact-hash>code').innerText().then(t=>/^[a-f0-9]{64}$/.test(t)),'Full SHA-256 fingerprint');
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Fingerprint overflow');
  await page.locator('.evidence-record').scrollIntoViewIfNeeded();await screenshot(page,'coverage-'+width);
 }
 await page.setViewportSize({width:1440,height:1000});await page.goto(base+'/fr/architecture');
 check(await page.locator('.engine-card .mechanism-svg').count()===10,'Distinct diagrams');
 await page.locator('.engine-card').first().scrollIntoViewIfNeeded();await screenshot(page,'mechanisms');
 for(const locale of ['fr','en']){
  await page.setViewportSize({width:390,height:844});
  await page.goto(base+'/'+locale+'/open/source');
  const sourceSearch=page.locator('#source-search');
  check(await sourceSearch.isVisible(),'Source search');
  const modules=await page.locator('[data-source-module]').count();
  await sourceSearch.fill('core/world.py');
  check(await page.locator('[data-source-module]:visible').count()===1,'Source path match');
  await sourceSearch.fill('not-a-wce-module');
  check(await page.locator('.inventory-search .search-empty').isVisible(),'Source empty state');
  await sourceSearch.fill('');
  check(await page.locator('[data-source-module]').count()===modules,'Source count preserved');
  check(await page.locator('.inventory-group[open]').count()===0,'Source disclosure restored');
  await sourceSearch.fill('World');
  check(await page.locator('[data-source-module]:visible').count()>0,'Symbol search');
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Source search overflow');
  await page.locator('.backend-inventory').scrollIntoViewIfNeeded();await screenshot(page,'source-search-'+locale);
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});
   for(const route of ['primitives','world-state','planning','language','measurement','evidence','certificates','learning','discovery','branching']){
    await page.goto(base+'/'+locale+'/architecture/'+route);
    check(await page.locator('.mechanism-scope').count()===1,'Evidence scope '+route);
    check(await page.locator('.scope-grid article').count()===3,'Scope levels '+route);
    check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Dossier overflow '+route+' '+width);
    renders++;
   }
  }
  await page.goto(base+'/'+locale+'/architecture');
  const search=page.locator('#mechanism-search');
  check(await search.isVisible(),'Atlas search visible');
  await search.fill('core/branching.py');
  check(await page.locator('[data-mechanism]:visible').count()===1,'Module search');
  await search.fill('not-a-wce-module');
  check(await page.locator('[data-mechanism]:visible').count()===0,'Empty search');
  check(await page.locator('.search-empty').isVisible(),'Empty state');
  await search.fill('');
  check(await page.locator('[data-mechanism]:visible').count()===10,'Clear search');
  check(await page.locator('.vocabulary details').count()===6,'Glossary size');
  const summary=page.locator('.vocabulary summary').first();
  await summary.focus();await page.keyboard.press('Enter');
  check(await page.locator('.vocabulary details').first().getAttribute('open')!==null,'Keyboard glossary');
  await page.goto(base+'/'+locale+'/');
  check(await page.locator('.understanding-grid article').count()===3,'Reading paths');
  await page.locator('.understanding').scrollIntoViewIfNeeded();await screenshot(page,'reading-paths-'+locale);
 }
 await verifyLaboratory();
 await page.setViewportSize({width:1440,height:1000});await page.goto(base+'/fr/architecture/planning');
 const heading=page.locator('[data-section="architecture"] .nav-heading');
 const toggle=page.locator('[data-section="architecture"] .nav-toggle');
 await heading.focus();check(await toggle.getAttribute('aria-expanded')==='true','Keyboard submenu');
 await page.keyboard.press('Escape');check(await toggle.getAttribute('aria-expanded')==='false','Close submenu with Escape');
 await heading.hover();check(await toggle.getAttribute('aria-expanded')==='true','Hover submenu');
 await page.keyboard.press('Escape');
 await page.setViewportSize({width:390,height:844});await page.goto(base+'/fr/');
 await page.locator('.menu').click();check(await page.locator('#navigation').isVisible(),'Mobile navigation');
 await page.locator('[data-section="architecture"] .nav-toggle').click();
 check(await page.locator('#nav-architecture').isVisible(),'Mobile architecture subpages');
 await page.keyboard.press('Escape');await page.keyboard.press('Escape');
 check(!await page.locator('#navigation').isVisible(),'Close mobile menu');
 await page.locator('footer').scrollIntoViewIfNeeded();
 check(await page.locator('.footer-directory').isVisible(),'Footer navigation');
 check(await page.locator('.footer-directory>div').count()===5,'Five footer sections');
 check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Footer overflow');
 await screenshot(page,'footer-reviewed');
 check(errors.length===0,'JS errors: '+errors.join('\n'));
 console.log('PASS: '+renders+' bilingual desktop/tablet/mobile renders; configuration operations and invariant; 9 metrics × 9 methods; intervals/tables; 10 verdicts; 5 source groups; themes; 10 mechanism diagrams; no overflow or JS errors');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(gateway)gateway.stdin.end();});
