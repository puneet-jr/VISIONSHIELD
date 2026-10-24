const fs=require('fs'),vm=require('vm'),assert=require('assert');
(async()=>{
 let now=0,next=0,tasks=new Map(),sent=0,maximum=0,released=0;
 const elements={};
 for(const id of ['start','stop','status','stats','facing','quality','preview','capture'])elements[id]={disabled:false,textContent:'',value:id==='quality'?'640':'environment'};
 Object.assign(elements.preview,{videoWidth:640,videoHeight:480,play:async()=>{}});
 Object.assign(elements.capture,{getContext:()=>({drawImage(){}}),toBlob:callback=>callback({size:4000})});
 const later=(fn,ms=0)=>{const id=++next;tasks.set(id,{at:now+ms,fn});return id};
 class Socket {
  static OPEN=1;
  constructor(){this.readyState=1;this.bufferedAmount=0;Socket.last=this;}
  send(){sent++;maximum=Math.max(maximum,vm.runInContext('pending.length',context));later(()=>this.onmessage?.({data:'{"ok":true}'}),500);}
  close(code=1000){this.readyState=3;this.onclose?.({code});}
 }
 const context=vm.createContext({document:{getElementById:id=>elements[id]},window:{addEventListener(){}},location:{hash:'',protocol:'https:',host:'example.test'},history:{replaceState(){}},URLSearchParams,isSecureContext:true,
  navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop(){released++}}]})}},WebSocket:Socket,performance:{now:()=>now},setTimeout:later,clearTimeout:id=>tasks.delete(id),console});
 vm.runInContext(fs.readFileSync('app/web/static/js/mobile.js','utf8'),context);
 await elements.start.onclick();Socket.last.onopen();
 while(now<1100){let entry=[...tasks.entries()].sort((a,b)=>a[1].at-b[1].at)[0];if(!entry||entry[1].at>1100)break;tasks.delete(entry[0]);now=entry[1].at;entry[1].fn();}
 assert(maximum<=3,'Unbounded sender window');assert(sent>=6,'Sender still waits one RTT per frame');
 Socket.last.onclose({code:4000});await Promise.resolve();await Promise.resolve();
 assert(released===1,'Camera tracks were not released on source change');
 assert(elements.status.textContent.includes('Source changed'));
 console.log(JSON.stringify({simulated_rtt_ms:500,frames_sent:sent,max_in_flight:maximum,source_change_releases_camera:true}));
})().catch(error=>{console.error(error);process.exit(1)});
