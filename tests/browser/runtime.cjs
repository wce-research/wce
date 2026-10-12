/* HTTPS proxy contract test: browser requests are intercepted and relayed to loopback only.
 * No certificate, public DNS, proxy deployment or Internet request is created. */
const {chromium}=require('playwright');
const {spawn,execFileSync}=require('child_process');
const http=require('http');
let server,browser;
const check=(value,message)=>{if(!value)throw Error(message);};
async function start(){return new Promise((resolve,reject)=>{
 server=spawn('make',['--no-print-directory','frontend-serve'],{detached:true,env:{...process.env,WCE_MODE:'production',WCE_PUBLIC_ORIGIN:'https://wce.test',WCE_HOST:'127.0.0.1',WCE_PORT:'0'},stdio:['ignore','pipe','pipe']});
 const timer=setTimeout(()=>reject(Error('Production fixture not ready')),15000);
 server.stderr.on('data',data=>{const match=data.toString().match(/WCE listening on 127\.0\.0\.1:(\d+)/);if(match){clearTimeout(timer);resolve(Number(match[1]));}});
 server.on('error',reject);server.on('exit',code=>{if(code){clearTimeout(timer);reject(Error('Fixture exited '+code));}});
});}
function relay(port,request){return new Promise((resolve,reject)=>{
 const url=new URL(request.url());
 const headers={...request.headers(),host:'wce.test'};
 const transport=http.request({hostname:'127.0.0.1',port,path:url.pathname+url.search,method:request.method(),headers},response=>{
  const body=[];response.on('data',chunk=>body.push(chunk));response.on('end',()=>resolve({status:response.statusCode,headers:response.headers,body:Buffer.concat(body)}));
 });transport.on('error',reject);transport.end(request.postDataBuffer());
});}
(async()=>{
 const port=await start();browser=await chromium.launch({headless:true,args:process.env.WCE_BROWSER_NO_SANDBOX==='1'?['--no-sandbox']:[]});
 const context=await browser.newContext();
 await context.route('**/*',async route=>{
  if(new URL(route.request().url()).hostname!=='wce.test')return route.abort();
  const response=await relay(port,route.request());
  // Playwright requires scalar values; preserve Set-Cookie from the production runtime.
  const headers={};for(const [key,value] of Object.entries(response.headers))if(!['connection','transfer-encoding'].includes(key))headers[key]=Array.isArray(value)?value.join('\n'):String(value);
  await route.fulfill({...response,headers});
 });
 const page=await context.newPage();const errors=[];page.on('pageerror',error=>errors.push(error.message));
 for(const locale of ['en','fr']){
  await page.goto('https://wce.test/'+locale+'/lab');
  await page.locator('[data-command]').first().click();await page.locator('#send').click();
  await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  check((await page.locator('#history .entry').first().innerText()).includes('OK.'),'HTTPS lab action '+locale);
  await page.locator('[data-command]').nth(1).click();await page.locator('#send').click();
  await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  check((await page.locator('#history .entry').first().innerText()).includes('2 m'),'HTTPS session continuity '+locale);
 }
 const cookie=(await context.cookies('https://wce.test')).find(item=>item.name==='wce_session');
 check(cookie&&cookie.secure&&cookie.httpOnly&&cookie.sameSite==='Strict','Production browser cookie flags');
 check(cookie.expires>Date.now()/1000+3500,'Explicit production cookie lifetime');
 check(!(await page.evaluate(()=>document.cookie)).includes('wce_session'),'Session token hidden from JavaScript');
 check(errors.length===0,'No browser errors');
 console.log('PASS: intercepted HTTPS proxy contract; FR/EN lab continuity; Secure/HttpOnly/Strict/expiry; no public network or certificate');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
 // Signal the owned Python recipe directly: some managed process environments
 // reject group signals. Do not silently leave the Make child/server running.
 if(server&&server.exitCode===null){
  const children=execFileSync('ps',['--ppid',String(server.pid),'-o','pid='],{encoding:'utf8'}).trim().split(/\s+/).filter(Boolean);
  for(const pid of children)process.kill(Number(pid),'SIGTERM');
  await new Promise((resolve,reject)=>{
   const timeout=setTimeout(()=>reject(Error('Proxy fixture did not stop gracefully')),10000);
   server.once('exit',()=>{clearTimeout(timeout);resolve();});
  });
 }
 if(browser)await browser.close();
});
