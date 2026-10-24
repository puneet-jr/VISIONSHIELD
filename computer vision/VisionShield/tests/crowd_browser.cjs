const fs=require('fs'),path=require('path'),{spawn}=require('child_process');
(async()=>{
 const chrome=process.env.CHROME_PATH||path.join(process.env.LOCALAPPDATA,'ms-playwright/chromium-1228/chrome-win64/chrome.exe');
 const browser=spawn(chrome,['--headless','--disable-gpu','--no-first-run','--user-data-dir='+path.resolve('artifacts/crowd-browser-profile'),'--remote-debugging-port=9224','about:blank'],{windowsHide:true,stdio:'ignore'});
 let ws;const sockets=[];let backup=[];
 const local=process.env.BASE_URL||'http://localhost:8000';
 try{
  for(let i=0;i<40;i++){try{await fetch('http://127.0.0.1:9224/json');break}catch{await new Promise(r=>setTimeout(r,250));}}
  const targets=await(await fetch('http://127.0.0.1:9224/json')).json();ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  let seq=0;const pending=new Map(),errors=[];
  ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,250));}throw Error('Timeout '+expression);};
  await call('Runtime.enable');await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  await fetch(local+'/api/source',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:'phone_upload'})});
  // Dedicated CLI WebSocket clients imitate separate devices; browser remains the viewer.

  const fixture=fs.readFileSync('data/pose-smoke-bus.jpg');
  for(const device of ['d'.repeat(32),'e'.repeat(32)]){
   const camera=new WebSocket(local.replace('http','ws')+'/ws/camera',{headers:{'Cookie':'vs_device='+device}});
   // Node native WebSocket does not support arbitrary handshake headers; local operator trust
   // supplies authorization, and missing device cookies produce independent session UUIDs.
   sockets.push(camera);await new Promise((resolve,reject)=>{camera.addEventListener('open',resolve,{once:true});camera.addEventListener('error',reject,{once:true});});
   camera.addEventListener('message',()=>{if(camera.readyState===WebSocket.OPEN)setTimeout(()=>{if(camera.readyState===WebSocket.OPEN)camera.send(fixture);},220);});camera.send(fixture);
  }
  await call('Page.navigate',{url:local+'/'});
  await wait("document.getElementById('overlayMode')?.value==='crowd' && document.getElementById('videoElement').naturalWidth>0");
  await evaluate("window.labelRectangles=0;const ctx=document.getElementById('trackingCanvas').getContext('2d'),original=ctx.fillRect.bind(ctx);ctx.fillRect=(...args)=>{window.labelRectangles++;return original(...args)}");
  await new Promise(r=>setTimeout(r,1000));
  if(await evaluate('window.labelRectangles!==0'))throw Error('Crowd mode draws label rectangles');
  fs.writeFileSync('artifacts/clean-crowd-dashboard.png',Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  await evaluate("document.getElementById('overlayMode').value='details';document.getElementById('overlayMode').dispatchEvent(new Event('change'))");
  await wait('window.labelRectangles>0');
  await call('Page.navigate',{url:local+'/guard'});
  await wait("document.querySelectorAll('.camera .picture img[src]').length>=2 && document.getElementById('cameraCount').textContent==='2'");
  await wait("document.querySelectorAll('.camera button:not([hidden])').length>=2");
  await evaluate("document.querySelector('.camera button').click()");await wait("document.getElementById('zoneDialog').open");
  const before=await(await fetch(local+'/api/streams')).json();const first=before.streams.find(s=>s.online);backup=first.zones;
  await evaluate("document.querySelector('[name=zoneName]').value='BROWSER TEST crowd';document.querySelector('[name=max_capacity]').value='2';document.getElementById('zoneForm').requestSubmit()");
  await wait("!document.getElementById('zoneDialog').open");
  await wait("[...document.querySelectorAll('.zone')].some(n=>n.textContent.includes('CHECK:'))");
  const desktop=await evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth,cameras:document.querySelectorAll('.camera').length,warnings:document.getElementById('warningCount').textContent,errors:document.getElementById('error').textContent})");
  fs.writeFileSync('artifacts/crowd-desktop.png',Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  const mobile=await evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth})");
  fs.writeFileSync('artifacts/crowd-mobile.png',Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  if(desktop.scroll>desktop.width||mobile.scroll>mobile.width||errors.length||desktop.errors)throw Error(JSON.stringify({desktop,mobile,errors}));
  const report={passed:true,desktop,mobile,zone_form:'per-camera save and real detection capacity warning passed',crowd_overlay:'default dot map draws no label rectangles; detailed mode restores labels',fixture:'Two separate socket sessions streaming the same static official bus photo; not physical phones',errors};
  fs.writeFileSync('artifacts/crowd-browser-verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  await fetch(local+'/api/streams/'+first.id+'/zones',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zones:backup})});
  await call('Browser.close');
 }finally{for(const camera of sockets)camera.close();ws?.close();browser.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
