/* Bounded camera transport: newest frames, at most three unacknowledged uploads. */
const $ = id => document.getElementById(id);
let media, socket, wake;
let running = false, encoding = false, runId = 0, retries = 0;
let pumpTimer, reconnectTimer, sent = 0, started = 0, pending = [], roundtrip = 0;
const status = message => { $('status').textContent = message; };
const setControls = active => {
    $('start').disabled = active; $('stop').disabled = !active;
    $('facing').disabled = active; $('quality').disabled = active;
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
    canvas.width = Math.min(Number($('quality').value), video.videoWidth);
    canvas.height = Math.round(canvas.width * video.videoHeight/video.videoWidth);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    encoding = true;
    canvas.toBlob(blob => {
        if (runId === version) encoding = false;
        if (!blob || !running || runId !== version || socket !== active || active.readyState !== WebSocket.OPEN) return;
        if (blob.size > 512000 || pending.length >= 3 || active.bufferedAmount > 220000) return;
        pending.push(performance.now());
        active.send(blob);
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
            const message = event.code === 4000 ? 'Source changed on dashboard. Camera stopped.' : event.code === 1013 ? 'Another phone is connected. Stop it first.' : 'Pairing expired. Scan the QR code again.';
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
        setControls(true); await pair();
        const width = Number($('quality').value);
        media = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:$('facing').value},width:{ideal:width},height:{ideal:Math.round(width*.75)},frameRate:{ideal:20,max:24}},audio:false});
        $('preview').srcObject = media; await $('preview').play();
        running = true; runId++; sent = 0; roundtrip = 0; started = performance.now();
        try { wake = await navigator.wakeLock?.request('screen'); } catch {}
        connect();
    } catch (error) { await stop(); status(error.message); }
};
$('stop').onclick = stop;
window.addEventListener('pagehide', stop);
if (!isSecureContext) status('HTTPS required: use the secure pairing URL from your dashboard.');
