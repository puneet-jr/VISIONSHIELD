/* Bounded camera transport: newest frames, at most three unacknowledged uploads. */
const $ = id => document.getElementById(id);
let media, socket, wake;
let running = false, encoding = false, runId = 0, retries = 0;
let pumpTimer, reconnectTimer, sent = 0, started = 0, pending = [], roundtrip = 0;
let clockOffset = 0;
async function syncClock() {
    try { const before=Date.now(); const response=await fetch('/api/time'); const result=await response.json(); clockOffset=result.server_time_ms-(before+Date.now())/2; }
    catch { clockOffset=0; }
}
const status = message => { $('status').textContent = message; };
const setControls = active => {
    $('start').disabled = active; $('stop').disabled = !active;
    $('facing').disabled = false; $('quality').disabled = active;
    $('shareLocation').disabled = !active;
};

async function pair() {
    const token = new URLSearchParams(location.hash.slice(1)).get('token');
    if (!token) return;
    const response = await fetch('/api/pair', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({token})});
    if (!response.ok) throw Error('Pairing expired. Scan a new QR code.');
    history.replaceState(null, '', location.pathname);
}

function pump() {
    if (!running) return;
    pumpTimer = setTimeout(pump, 1000/15);
    const active = socket, version = runId;
    if (!active || active.readyState !== WebSocket.OPEN) return;
    if (pending.length && performance.now() - pending[0] > 8000) { active.close(); return; }
    if (encoding || pending.length >= 3 || active.bufferedAmount > 220000) return;
    const video = $('preview'), canvas = $('capture');
    if (!video.videoWidth) return;
    const angle=Number($('rotation')?.value||0),swapped=angle%180!==0;
    const sourceWidth=swapped?video.videoHeight:video.videoWidth,sourceHeight=swapped?video.videoWidth:video.videoHeight;
    const scale=Math.min(1,Number($('quality').value)/Math.max(sourceWidth,sourceHeight));
    const targetWidth=Math.round(sourceWidth*scale),targetHeight=Math.round(sourceHeight*scale);
    if(canvas.width!==targetWidth||canvas.height!==targetHeight){canvas.width=targetWidth;canvas.height=targetHeight;}
    const ctx=canvas.getContext('2d');
    if(angle){ctx.save();ctx.translate(canvas.width/2,canvas.height/2);ctx.rotate(angle*Math.PI/180);ctx.drawImage(video,-video.videoWidth*scale/2,-video.videoHeight*scale/2,video.videoWidth*scale,video.videoHeight*scale);ctx.restore();}
    else ctx.drawImage(video,0,0,canvas.width,canvas.height);
    const capturedAt=(Date.now()+clockOffset)/1000;
    encoding = true;
    canvas.toBlob(blob => {
        if (runId === version) encoding = false;
        if (!blob || !running || runId !== version || socket !== active || active.readyState !== WebSocket.OPEN) return;
        if (blob.size > 512000 || pending.length >= 3 || active.bufferedAmount > 220000) return;
        pending.push(performance.now());
        const header=new ArrayBuffer(12), view=new DataView(header);
        new Uint8Array(header).set([86,83,67,49]); // VSC1 + server-aligned capture time.
        view.setFloat64(4,capturedAt,false);
        active.send(new Blob([header,blob],{type:'application/octet-stream'}));
    }, 'image/jpeg', Number($('quality').value) > 640 ? .78 : .65);
}

function connect() {
    if (!running) return;
    status('Connecting securely...');
    const active = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/camera`);
    socket = active;
    active.onopen = () => {
        if (!running || socket !== active) { active.close(); return; }
        pending = []; retries = 0; status('Live - streaming to VisionShield');
        clearTimeout(pumpTimer); pump();
    };
    active.onmessage = event => {
        if (socket !== active) return;
        const result = JSON.parse(event.data);
        if (result.error) { stop().then(() => status(result.error)); return; }
        const acknowledged = pending.shift();
        if (acknowledged !== undefined) roundtrip = .8*roundtrip + .2*(performance.now()-acknowledged);
        sent++;
        $('stats').textContent = `${(sent/Math.max(1,(performance.now()-started)/1000)).toFixed(1)} uploads/s | ${$('capture').width} x ${$('capture').height} | RTT ${Math.round(roundtrip)} ms | ${pending.length}/3 in flight`;
    };
    active.onerror = () => status('Connection interrupted. Checking network...');
    active.onclose = event => {
        if (!running || socket !== active) return;
        clearTimeout(pumpTimer); pending = [];
        if ([1008,1013,4000].includes(event.code)) {
            const message = event.code === 4000 ? 'Source changed on dashboard. Restart this camera.' : event.code === 1013 ? (event.reason || 'This browser is already streaming, or the camera limit is reached. Close duplicate tabs and retry.') : 'Pairing expired. Scan the QR code again.';
            stop().then(() => status(message)); return;
        }
        status('Reconnecting...');
        reconnectTimer = setTimeout(connect, Math.min(1000*2**retries++,10000));
    };
}

async function stop() {
    running = false; encoding = false; runId++;
    clearTimeout(pumpTimer); clearTimeout(reconnectTimer);
    socket?.close(); socket = null; pending = [];
    media?.getTracks().forEach(track => track.stop());
    $('preview').srcObject = null;
    setControls(false); status('Stopped - camera released');
    try { await wake?.release(); } catch {}
}

$('start').onclick = async () => {
    try {
        if (!isSecureContext || !navigator.mediaDevices) throw Error('Camera access requires HTTPS. Start VisionShield with --remote and scan its QR code.');
        setControls(true); await pair(); await syncClock();
        const width = Number($('quality').value);
        media = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:$('facing').value},width:{ideal:width},height:{ideal:Math.round(width*.75)},frameRate:{ideal:20,max:24}},audio:false});
        $('preview').srcObject = media; await $('preview').play();
        running = true; runId++; sent = 0; roundtrip = 0; started = performance.now();
        try { wake = await navigator.wakeLock?.request('screen'); } catch {}
        connect();
    } catch (error) { await stop(); status(error.message); }
};
$('stop').onclick = stop;
$('facing').onchange=async()=>{if(running){await stop();setTimeout(()=>$('start').click(),750);}};
if($('rotation'))$('rotation').onchange=async()=>{if(running){await stop();setTimeout(()=>$('start').click(),750);}};
$('shareLocation').onclick=()=>{
    if(!running)return;
    if(!navigator.geolocation){$('locationStatus').textContent='GPS is unavailable in this browser.';return;}
    $('locationStatus').textContent='Requesting camera-device location permission…';
    navigator.geolocation.getCurrentPosition(async position=>{
        if(!running)return;
        const payload={latitude:position.coords.latitude,longitude:position.coords.longitude,accuracy_m:position.coords.accuracy};
        try{const response=await fetch('/api/camera/location',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!response.ok)throw Error('Camera GPS update rejected ('+response.status+'). Restart streaming or re-pair.');$('locationStatus').textContent=`Camera GPS shared: ${payload.latitude.toFixed(6)}, ${payload.longitude.toFixed(6)} · accuracy ±${Math.round(payload.accuracy_m)} m. Use this button again after moving.`;}
        catch(error){$('locationStatus').textContent=error.message;}
    },error=>{$('locationStatus').textContent='GPS not shared: '+error.message+'. Video continues without location.';},{enableHighAccuracy:true,timeout:15000,maximumAge:10000});
};
window.addEventListener('pagehide', stop);
if (!isSecureContext) status('HTTPS required: use the secure pairing URL from your dashboard.');
