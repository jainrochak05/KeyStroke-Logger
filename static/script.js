let downTime = {};
let dwellTimes = [];
let registerSamples = [];
const REQUIRED_ATTEMPTS = 5;

document.addEventListener("DOMContentLoaded", () => {
  const box = document.getElementById("typingBox");
  if (!box) return;

  box.addEventListener("keydown", e => {
    if (e.key.length === 1) {
      downTime[e.key] = performance.now();
    }
    if (e.key === "Enter") {
      e.preventDefault();
    }
  });

  box.addEventListener("keyup", e => {
    if (e.key.length === 1 && downTime[e.key]) {
      let dwell = performance.now() - downTime[e.key];
      dwellTimes.push(dwell);
    }
  });
});

function resetTyped() {
  dwellTimes = [];
  downTime = {};
  const box = document.getElementById("typingBox");
  if (box) box.value = "";
}

// ---------- Registration flow: 5 attempts ----------

function registerAttempt() {
  const username = document.getElementById("reg_username")?.value.trim();
  const password = document.getElementById("reg_password")?.value;

  if (!username || !password) {
    alert("Enter username and password first.");
    return;
  }
  if (dwellTimes.length === 0) {
    alert("Type the password in the box before saving.");
    return;
  }

  const typed = document.getElementById("typingBox").value;
  if (typed !== password) {
    if (!confirm("Typed text does not match password. Save anyway?")) {
      resetTyped();
      return;
    }
  }

  registerSamples.push([...dwellTimes]);
  resetTyped();

  const info = document.getElementById("attemptInfo");
  if (info) info.innerText = `Attempts saved: ${registerSamples.length} / ${REQUIRED_ATTEMPTS}`;

  if (registerSamples.length < REQUIRED_ATTEMPTS) {
    alert("Attempt saved. Type again.");
    return;
  }

  // send all attempts to backend
  fetch("/register_data", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      username: username,
      password: password,
      samples: registerSamples
    })
  })
  .then(r => r.json())
  .then(d => {
    if (d.status === "registered") {
      alert("Registration complete! Model updated.");
      registerSamples = [];
      if (info) info.innerText = `Attempts saved: 0 / ${REQUIRED_ATTEMPTS}`;
    } else {
      alert("Error: " + (d.message || JSON.stringify(d)));
    }
  })
  .catch(() => alert("Registration failed (network error)."));
}


// ---------- Login flow ----------

function verifyUser() {
  const username = document.getElementById("login_username")?.value.trim();
  const password = document.getElementById("login_password")?.value;

  if (!username || !password) {
    alert("Enter username and password.");
    return;
  }
  if (dwellTimes.length === 0) {
    alert("Type your password in the box before clicking Login.");
    return;
  }

  fetch("/verify", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      username: username,
      password: password,
      dwell_times: dwellTimes
    })
  })
  .then(r => r.json())
  .then(d => {
    if (d.result === "success") {
      alert("Access Granted!");
    } else {
      alert("Access Denied! " + (d.predicted || d.reason));
    }
    resetTyped();
  })
  .catch(() => {
    alert("Login failed (network error).");
    resetTyped();
  });
}
