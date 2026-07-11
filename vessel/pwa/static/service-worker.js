const CACHE = "vessel-shell-v39";
const ASSETS = [
  "/pwa/app.js",
  "/pwa/styles.css",
  "/pwa/manifest.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Page asks the SW to take control via postMessage when the user hits
// refresh. The `controllerchange` listener in app.js then reloads so the
// freshly-fetched JS / CSS is what runs.
self.addEventListener("message", (event) => {
  if (event.data === "SKIP_WAITING") self.skipWaiting();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Never intercept the API — needs fresh state and auth.
  if (url.pathname.startsWith("/api/")) return;
  if (req.method !== "GET") return;

  // HTML / navigations: network-first so UI updates land on reload.
  const isNav =
    req.mode === "navigate" ||
    url.pathname === "/pwa/" ||
    url.pathname === "/pwa/index.html";
  if (isNav) {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          if (resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE).then((c) => c.put("/pwa/index.html", clone));
          }
          return resp;
        })
        .catch(() => caches.match("/pwa/index.html"))
    );
    return;
  }

  // Static assets under /pwa/: network-first too, with cache only as an
  // offline fallback. Stale-while-revalidate served the old JS/CSS on
  // every refresh and made iOS installs feel stuck on outdated UI —
  // network-first guarantees a successful refresh ships the latest
  // assets in one round trip.
  if (url.pathname.startsWith("/pwa/")) {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          if (resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE).then((c) => c.put(req, clone));
          }
          return resp;
        })
        .catch(() => caches.match(req))
    );
  }
});
