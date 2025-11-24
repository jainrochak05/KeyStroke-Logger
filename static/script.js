// signature drawing capture (supports mouse)
let mode = null;
let drawing = false;
let points = []; // current stroke points
let savedCount = 0;

function initDrawer(m) {
  mode = m;
  const canvas = document.getElementById("drawCanvas");
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#000";

  canvas.addEventListener("mousedown", e=>{
    drawing = true;
    points = [];
    ctx.clearRect(0,0,canvas.width,canvas.height);
    addPoint(e, canvas);
  });
  canvas.addEventListener("mousemove", e=>{
    if(!drawing) return;
    addPoint(e, canvas);
    draw(ctx);
  });
  window.addEventListener("mouseup", e=>{
    if(!drawing) return;
    drawing = false;
    addPoint(e, canvas);
    draw(ctx);
  });
  canvas.addEventListener("mouseleave", ()=>{ drawing=false; });
}

function addPoint(e, canvas){
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const t = performance.now();
  points.push({x, y, t});
}

function draw(ctx){
  if(points.length<2) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for(let i=1;i<points.length;i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.stroke();
}

function clearCanvas(){
  const c = document.getElementById("drawCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);
  points = [];
}

function saveSample(){
  const username = document.getElementById("reg_username").value.trim();
  const password = document.getElementById("reg_password").value;
  if(!username||!password){ alert("enter username & pass"); return;}
  if(!points || points.length<5){ alert("draw signature first"); return;}
  fetch("/register_sample", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({username, password, points})
  }).then(r=>r.json()).then(d=>{
    if(d.status==="saved"){ savedCount = d.user_samples; document.getElementById("saved").innerText="Saved: "+savedCount; alert("Saved"); }
    else alert("Error: "+(d.message||JSON.stringify(d)));
    clearCanvas();
  }).catch(()=>{alert("network error"); clearCanvas();});
}

function trainModel(){
  fetch("/train").then(r=>r.json()).then(d=>{
    if(d.status==="trained") alert("Trained for: "+d.users.join(", "));
    else alert("Train error: "+(d.message||JSON.stringify(d)));
  }).catch(()=>alert("train network error"));
}

function verifyUser(){
  const username = document.getElementById("login_username").value.trim();
  const password = document.getElementById("login_password").value;
  if(!username||!password){ alert("enter username & pass"); return;}
  if(!points || points.length<5){ alert("draw signature first"); return;}
  fetch("/verify", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({username, password, points})
  }).then(r=>r.json()).then(d=>{
    if(d.result==="success"){ alert("Access Granted! (conf="+(d.confidence||"n/a")+")"); }
    else alert("Access Denied! "+(d.predicted||d.reason));
    clearCanvas();
  }).catch(()=>{alert("network error"); clearCanvas();});
}
