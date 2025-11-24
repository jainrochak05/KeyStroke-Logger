let drawing = false;
let points = []; // current stroke points
let mode = null; // "register" or "login"

function initDrawingMode(m) {
  mode = m;
  const canvas = document.getElementById("drawCanvas");
  const ctx = canvas.getContext("2d");

  canvas.addEventListener("mousedown", e => {
    drawing = true;
    points = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    addPoint(e, canvas);
  });

  canvas.addEventListener("mousemove", e => {
    if (!drawing) return;
    addPoint(e, canvas);
    drawStroke(ctx);
  });

  canvas.addEventListener("mouseup", e => {
    if (!drawing) return;
    drawing = false;
    addPoint(e, canvas);
    drawStroke(ctx);
  });

  canvas.addEventListener("mouseleave", () => {
    if (drawing) {
      drawing = false;
    }
  });
}

function addPoint(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const t = performance.now();
  points.push({ x, y, t });
}

function drawStroke(ctx) {
  if (points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
}

function resetCanvas() {
  const canvas = document.getElementById("drawCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  points = [];
}

// ---------------- Registration ---------------- //

function saveRegisterSample() {
  const username = document.getElementById("reg_username")?.value.trim();
  const password = document.getElementById("reg_password")?.value;

  if (!username || !password) {
    alert("Enter username and password first.");
    return;
  }

  if (!points || points.length < 5) {
    alert("Draw a circle first.");
    return;
  }

  fetch("/register_sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, points })
  })
    .then(r => r.json())
    .then(d => {
      if (d.status === "saved") {
        alert("Sample saved!");
        const info = document.getElementById("sampleInfo");
        if (info && d.user_samples !== undefined) {
          info.innerText = "Samples saved: " + d.user_samples;
        }
      } else {
        alert("Error: " + (d.message || JSON.stringify(d)));
      }
      resetCanvas();
    })
    .catch(() => {
      alert("Network error while saving sample.");
      resetCanvas();
    });
}

function trainModel() {
  fetch("/train")
    .then(r => r.json())
    .then(d => {
      if (d.status === "trained") {
        alert("Model trained for users: " + d.users.join(", "));
      } else {
        alert("Train error: " + (d.message || JSON.stringify(d)));
      }
    })
    .catch(() => alert("Network error while training model."));
}

// ---------------- Login ---------------- //

function verifyUser() {
  const username = document.getElementById("login_username")?.value.trim();
  const password = document.getElementById("login_password")?.value;

  if (!username || !password) {
    alert("Enter username and password.");
    return;
  }
  if (!points || points.length < 5) {
    alert("Draw your circle before clicking Login.");
    return;
  }

  fetch("/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, points })
  })
    .then(r => r.json())
    .then(d => {
      if (d.result === "success") {
        alert("Access Granted!");
      } else {
        alert("Access Denied! " + (d.predicted || d.reason));
      }
      resetCanvas();
    })
    .catch(() => {
      alert("Network error during login.");
      resetCanvas();
    });
}
